"""本地 TTS adapter 的单测：切句逻辑 + 并行合成顺序 + 语速透传。

不打网络、不加载 327MB 权重：模型调用全部用假 pipeline 替身，
真机效果由 scripts/bench_tts_timeline.py 量化。
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import kokoro_tts_server as tts  # noqa: E402


class FakePipeline:
    """记录调用参数的假 pipeline：每个分片返回定长静音，长度按字数算便于验证顺序。"""

    sr = 24000

    def __init__(self) -> None:
        self.calls: list[tuple[str, float]] = []
        self.voices: dict[str, object] = {}

    def load_voice(self, voice: str):
        self.voices[voice] = object()
        return self.voices[voice]

    def __call__(self, text: str, voice: str | None = None, speed: float = 1):
        self.calls.append((text, speed))
        samples = np.zeros(int(self.sr * 0.1 * len(text)), dtype=np.float32)
        yield None, None, samples


def test_split_keeps_punctuation_and_splits_on_sentence_end():
    parts = tts.split_sentences("第一句话。第二句话！第三句话？")
    assert parts == ["第一句话。", "第二句话！", "第三句话？"]


def test_split_treats_newline_as_boundary():
    parts = tts.split_sentences("第一段内容在第一行。\n\n第二段内容在第二行。")
    assert all("\n" not in p for p in parts)
    assert len(parts) == 2


def test_split_merges_tiny_pieces_to_avoid_choppy_audio():
    # 单字短句和下一句合并，避免每两个字就切一次导致拼接生硬
    parts = tts.split_sentences("好。这句话稍微长一点点。", min_chars=6)
    assert len(parts) == 1
    assert parts[0].endswith("。")


def test_split_hard_wraps_long_sentence_at_max_chars():
    long = "，" .join(["这是一个很长的分句" for _ in range(8)]) + "。"
    parts = tts.split_sentences(long, max_chars=20)
    assert len(parts) > 1
    assert all(len(p) <= 20 for p in parts)
    assert "".join(parts).replace("，", "") == long.replace("，", "")


def test_split_returns_empty_for_blank_text():
    assert tts.split_sentences("   \n  ") == []


def test_synth_all_preserves_order_and_passes_speed():
    fake = FakePipeline()
    text = "第一句够长了吧。第二句也够长了。第三句同样够长。"
    audio, parts = tts.synth_all(fake, text, voice="v.pt", speed=1.2, workers=2)

    assert [c[0] for c in fake.calls] == parts, "并行必须按原顺序返回结果"
    assert all(c[1] == 1.2 for c in fake.calls), "speed 要透传给每个分片"
    expected = sum(int(fake.sr * 0.1 * len(p)) for p in parts)
    assert len(audio) == expected


def test_synth_all_single_chunk_when_text_is_short():
    fake = FakePipeline()
    audio, parts = tts.synth_all(fake, "只有一句话。", voice="v.pt", speed=1.0, workers=4)
    assert len(parts) == 1
    assert len(fake.calls) == 1
    assert len(audio) > 0


def test_synth_all_serial_mode_still_covers_every_chunk():
    """回归：workers=1 曾只合成第一片，其余分片被静默丢弃（音频短了一大截）。"""
    fake = FakePipeline()
    text = "第一句够长了吧。第二句也够长了。第三句同样够长。"
    audio, parts = tts.synth_all(fake, text, voice="v.pt", speed=1.0, workers=1)

    assert len(parts) == 3
    assert [c[0] for c in fake.calls] == parts, "串行模式也必须逐片合成"
    expected = sum(int(fake.sr * 0.1 * len(p)) for p in parts)
    assert len(audio) == expected


def test_resolve_voice_falls_back_to_id_when_local_pt_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(tts, "MODEL_DIR", tmp_path)
    assert tts.resolve_voice("zf_001") == "zf_001"


def test_resolve_voice_uses_local_pt_when_present(monkeypatch, tmp_path):
    monkeypatch.setattr(tts, "MODEL_DIR", tmp_path)
    (tmp_path / "voices").mkdir()
    (tmp_path / "voices" / "zf_001.pt").write_bytes(b"x")
    assert tts.resolve_voice("zf_001").endswith("zf_001.pt")


def test_local_weights_requires_both_files(monkeypatch, tmp_path):
    monkeypatch.setattr(tts, "MODEL_DIR", tmp_path)
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    assert tts.local_weights() is None, "只有 config 没有权重时必须当作没下完"
    (tmp_path / "kokoro-v1_1-zh.pth").write_bytes(b"x")
    assert tts.local_weights() is not None


def test_health_reports_speed_default():
    body = tts.health()
    assert body["status"] == "ok"
    assert float(body["speed"]) > 0


def _tone_then_silence(gap_s: float, sr: int = 24000) -> np.ndarray:
    t = np.arange(int(sr * 0.3)) / sr
    tone = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return np.concatenate([tone, np.zeros(int(sr * gap_s), dtype=np.float32), tone])


def test_cap_silence_shortens_overlong_pauses():
    sr = tts.SAMPLE_RATE
    audio = _tone_then_silence(1.0)  # 0.3 + 1.0 + 0.3 = 1.6s
    out = tts.cap_silence(audio, max_pause_ms=380)

    assert len(out) < len(audio), "1 秒的长停顿应该被削短"
    assert abs(len(out) / sr - (0.6 + 0.38)) < 0.06, "只削停顿，语音部分不能被吃掉"


def test_cap_silence_keeps_short_pauses_untouched():
    audio = _tone_then_silence(0.2)
    out = tts.cap_silence(audio, max_pause_ms=380)
    assert len(out) == len(audio), "短停顿属于正常语气，不能动"


def test_cap_silence_zero_disables_trimming():
    audio = _tone_then_silence(1.0)
    assert len(tts.cap_silence(audio, max_pause_ms=0)) == len(audio)
