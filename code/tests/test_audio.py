"""音频转写链路测试。

全部离线：不加载 900MB 模型、不联网。真实模型只在
`test_audio_integration.py` 那种带 skip 标记的用例里跑（需 --run-slow）。
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from feiyi.rag.audio import (
    AsrUnavailableError,
    AudioError,
    FunAsrEngine,
    Segment,
    StubAsrEngine,
    Transcript,
    audio_duration_ms,
    decode_audio,
    format_transcript,
    load_transcript,
    normalize_segments,
    parse_transcript,
    save_transcript,
    transcribe_audio,
    transcript_chunks,
)


# ---------------------------------------------------------------- 工具
def write_wav(path, seconds=2.0, sr=16000, channels=1, freq=440.0, amplitude=0.3):
    """写一个纯音 wav，用来测解码与重采样（不需要真实语音）。"""
    import soundfile as sf

    n = int(sr * seconds)
    t = np.arange(n) / sr
    tone = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    data = np.repeat(tone[:, None], channels, axis=1) if channels > 1 else tone
    sf.write(str(path), data, sr)
    return path


class EmptyEngine:
    """什么都不识别出来的引擎：用于验证「空结果必须告警」。"""

    name = "empty"

    def transcribe(self, samples, sample_rate=16000):
        return []


# ---------------------------------------------------------------- 解码
def test_decode_resamples_and_downmixes_to_16k_mono(tmp_path):
    """44.1k 立体声必须重采样成 16k 单声道，且时长不能变。

    直接拿 44.1k 字节喂模型不会报错，出来的是噪音——
    所以这里必须验采样率，而不是只验「能读出来」。
    """
    path = write_wav(tmp_path / "a.wav", seconds=1.5, sr=44100, channels=2)
    samples, sr = decode_audio(path)
    assert sr == 16000
    assert samples.ndim == 1
    assert abs(len(samples) / sr - 1.5) < 0.05
    assert samples.dtype == np.float32


def test_decode_rejects_missing_and_empty(tmp_path):
    with pytest.raises(AudioError, match="不存在"):
        decode_audio(tmp_path / "nope.wav")
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    with pytest.raises(AudioError, match="空的"):
        decode_audio(empty)


def test_decode_non_audio_file_raises_explicitly(tmp_path):
    """把 txt 当音频传进来，必须明确报错，不能返回空数组让上层当成静音。"""
    fake = tmp_path / "not_audio.wav"
    fake.write_text("这不是音频", encoding="utf-8")
    with pytest.raises(AudioError):
        decode_audio(fake)


def test_decode_flac_and_m4a(tmp_path):
    """PyAV 要能读常见格式——交付现场不会都给你 wav。"""
    import soundfile as sf

    flac = tmp_path / "a.flac"
    sf.write(str(flac), np.zeros(16000, dtype=np.float32), 16000)
    samples, sr = decode_audio(flac)
    assert sr == 16000 and len(samples) == 16000


def test_decode_keeps_both_channels(tmp_path):
    """立体声只取一个声道会丢掉一半信息；左右声道必须都能出来。

    PyAV 用的是等功率下混（-3dB 声像），单边满幅源下混后约 0.354，
    不是简单平均的 0.25——所以这里验的是「左右对称且都不为零」，
    而不是写死某个数值。
    """
    import soundfile as sf

    sr = 16000
    for name, left, right in [("L", 0.5, 0.0), ("R", 0.0, 0.5)]:
        sf.write(
            str(tmp_path / f"{name}.wav"),
            np.stack([np.full(sr, left, dtype=np.float32), np.full(sr, right, dtype=np.float32)], axis=1),
            sr,
        )
    rms_l = float(np.sqrt(np.mean(decode_audio(tmp_path / "L.wav")[0] ** 2)))
    rms_r = float(np.sqrt(np.mean(decode_audio(tmp_path / "R.wav")[0] ** 2)))
    assert rms_l > 0.2 and rms_r > 0.2, "某个声道被丢弃了"
    assert abs(rms_l - rms_r) < 0.02, "左右声道处理不对称"


def test_audio_duration_ms():
    assert audio_duration_ms(np.zeros(16000, dtype=np.float32), 16000) == 1000


# ---------------------------------------------------------------- 时间戳清洗
def test_normalize_segments_drops_empty_and_fixes_order():
    raw = [
        Segment(5000, 6000, "第三句"),
        Segment(0, 1000, "第一句"),
        Segment(2000, 2000, "第二句零时长"),
        Segment(3000, 3500, "   "),
    ]
    out = normalize_segments(raw, duration_ms=10000)
    assert [s.text for s in out] == ["第一句", "第二句零时长", "第三句"]
    assert out == sorted(out, key=lambda s: s.start_ms)


def test_normalize_segments_clamps_out_of_range():
    """模型偶尔给超出音频长度的时间戳，放任不管用户会跳到不存在的时刻。"""
    out = normalize_segments([Segment(-500, 999999, "太长")], duration_ms=8000)
    assert out[0].start_ms == 0
    assert out[0].end_ms == 8000


def test_normalize_segments_prevents_overlap_backwards():
    out = normalize_segments(
        [Segment(0, 5000, "A"), Segment(3000, 6000, "B")], duration_ms=10000
    )
    assert out[1].start_ms >= out[0].end_ms


def test_segment_duration_never_negative():
    assert Segment(5000, 1000, "x").duration_ms == 0


# ---------------------------------------------------------------- 桩引擎与整段转写
def test_stub_engine_splits_by_audio_length(tmp_path):
    path = write_wav(tmp_path / "a.wav", seconds=3.0)
    tr = transcribe_audio(path, engine=StubAsrEngine("测试文本"))
    assert tr.engine == "stub"
    assert tr.duration_ms == pytest.approx(3000, abs=60)
    assert tr.segments[0].text == "测试文本"
    assert tr.segments[0].start_ms == 0


def test_transcribe_warns_when_nothing_recognized(tmp_path):
    """整段没识别出内容时必须告警：直接入库等于零卡片，而程序毫无反应。"""
    path = write_wav(tmp_path / "silence.wav", seconds=2.0, amplitude=0.0)
    tr = transcribe_audio(path, engine=EmptyEngine())
    assert tr.segments == []
    assert any("未识别到" in w for w in tr.warnings)


def test_transcribe_warns_on_mostly_silent_audio(tmp_path):
    path = write_wav(tmp_path / "quiet.wav", seconds=2.0, amplitude=0.0)
    tr = transcribe_audio(path, engine=StubAsrEngine("有声"))
    assert any("静音" in w for w in tr.warnings)


def test_max_seconds_trims_for_trial_runs(tmp_path):
    """长录音先试跑前 N 秒，避免一小时的口述史一上来就等十分钟。"""
    path = write_wav(tmp_path / "long.wav", seconds=5.0)
    tr = transcribe_audio(path, engine=StubAsrEngine("片段"), max_seconds=2)
    assert tr.duration_ms == pytest.approx(2000, abs=60)
    assert any("max_seconds" in w for w in tr.warnings)


# ---------------------------------------------------------------- 切块
def _tr(segments, duration_ms=None):
    dur = duration_ms if duration_ms is not None else max((s.end_ms for s in segments), default=0)
    return Transcript(source="口述史.wav", duration_ms=dur, segments=segments, engine="stub")


def test_chunks_split_on_long_pause():
    """长停顿 = 换话题，必须切开；短停顿必须合并。"""
    tr = _tr(
        [
            Segment(0, 2000, "第一句讲学艺。"),
            Segment(2300, 4000, "第二句还是学艺。"),  # 300ms，合并
            Segment(8000, 10000, "这里停顿很久，换话题了。"),  # 4000ms，切开
        ]
    )
    chunks = transcript_chunks(tr, size=300, gap_ms=1200)
    assert len(chunks) == 2
    assert "第一句" in chunks[0].text and "第二句" in chunks[0].text
    assert "换话题" in chunks[1].text
    assert chunks[0].meta["start_ms"] == 0
    assert chunks[0].meta["end_ms"] == 4000


def test_chunks_split_on_size():
    tr = _tr([Segment(i * 1000, i * 1000 + 900, f"第{i}段" + "字" * 90) for i in range(8)])
    chunks = transcript_chunks(tr, size=300)
    assert len(chunks) > 1
    assert all(len(c.text) <= 320 for c in chunks)


def test_chunk_meta_carries_time_range():
    tr = _tr([Segment(65000, 68000, "讲碗碗腔的乐器。")])
    chunk = transcript_chunks(tr)[0]
    assert chunk.meta["kind"] == "audio"
    assert chunk.meta["start_ms"] == 65000
    assert chunk.meta["end_ms"] == 68000
    assert chunk.meta["duration_ms"] == 3000


def test_chunk_ids_unique_across_groups():
    """不同位置必须算出不同 id，否则入库被 store 静默去重（同 PDF 那条链路）。"""
    tr = _tr([Segment(i * 3000, i * 3000 + 1000, "同样的话") for i in range(6)])
    chunks = transcript_chunks(tr, gap_ms=1000)
    assert len(chunks) == 6
    assert len({c.id for c in chunks}) == 6


def test_multi_speaker_prefixes_and_splits():
    tr = _tr(
        [
            Segment(0, 2000, "您是哪一年开始学皮影的？", speaker="说话人1"),
            Segment(2100, 5000, "我十一岁那年跟着我爹学。", speaker="说话人2"),
        ]
    )
    chunks = transcript_chunks(tr, keep_speaker=True)
    assert len(chunks) == 2, "换人说话就该切一张新卡片"
    assert chunks[0].text.startswith("说话人1：")
    assert chunks[1].meta["speaker"] == "说话人2"


def test_single_speaker_gets_no_prefix_noise():
    tr = _tr([Segment(0, 2000, "我十一岁学皮影。", speaker="说话人1")])
    chunk = transcript_chunks(tr)[0]
    assert chunk.text == "我十一岁学皮影。"


def test_empty_transcript_yields_no_chunks():
    assert transcript_chunks(_tr([])) == []


# ---------------------------------------------------------------- 转写稿往返
def test_transcript_roundtrip_preserves_timestamps():
    tr = _tr(
        [
            Segment(0, 3000, "我叫王守义。"),
            Segment(3500, 8000, "今年六十七了。"),
            Segment(70000, 73000, "这门手艺不能断。"),
        ],
        duration_ms=75000,
    )
    text = format_transcript(tr)
    back = parse_transcript(text, source="校对稿")
    assert [s.text for s in back.segments] == [s.text for s in tr.segments]
    # 校对稿的时间戳是「秒」精度（给人改字用的，不是给机器对齐用的），
    # 往返必然丢亚秒部分，误差上限就是 1 秒。
    for got, want in zip(back.segments, tr.segments):
        assert abs(got.start_ms - want.start_ms) <= 1000
    # 末段结束时间靠文件头的总时长补回来，不能退化成 0
    assert back.segments[-1].end_ms == 75000


def test_parse_transcript_merges_lines_without_timestamp():
    """校对时手滑敲了回车，不能把内容丢掉或变成新的一段。"""
    text = "# 时长：00:10\n[00:00] 第一句前半\n后半句没写时间戳\n"
    tr = parse_transcript(text)
    assert len(tr.segments) == 1
    assert tr.segments[0].text == "第一句前半后半句没写时间戳"


def test_parse_transcript_handles_hours():
    tr = parse_transcript("[1:02:05] 一小时后的一句话\n")
    assert tr.segments[0].start_ms == (3600 + 2 * 60 + 5) * 1000


def test_parse_transcript_recovers_speaker():
    tr = parse_transcript("[00:01] 说话人2：我十一岁学的\n")
    assert tr.segments[0].speaker == "说话人2"
    assert tr.segments[0].text == "我十一岁学的"


def test_save_and_load_transcript_txt_and_jsonl(tmp_path):
    tr = _tr([Segment(0, 2000, "第一句。"), Segment(2000, 4000, "第二句。")], duration_ms=4000)
    txt = save_transcript(tr, tmp_path / "t.txt")
    jsonl = save_transcript(tr, tmp_path / "t.jsonl")

    assert "时长：00:04" in txt.read_text(encoding="utf-8")
    back_txt = load_transcript(txt)
    assert [s.text for s in back_txt.segments] == ["第一句。", "第二句。"]

    back_jsonl = load_transcript(jsonl)
    assert [s.start_ms for s in back_jsonl.segments] == [0, 2000]


def test_load_transcript_bad_jsonl_reports_line(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"start_ms":0,"text":"好"}\nnot json\n', encoding="utf-8")
    with pytest.raises(AudioError, match="第 2 行"):
        load_transcript(bad)


def test_transcript_stats_exposes_speech_ratio():
    tr = _tr([Segment(0, 5000, "说了一半")], duration_ms=10000)
    stats = tr.stats()
    assert stats["speech_ratio"] == 0.5
    assert stats["duration"] == "00:10"


# ---------------------------------------------------------------- 依赖与契约
def test_funasr_engine_reports_missing_dependency(monkeypatch):
    """依赖缺失必须抛异步可辨识的错误，接口层才能返回 503 并说明装什么。"""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "funasr":
            raise ImportError("no funasr")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    engine = FunAsrEngine()
    with pytest.raises(AsrUnavailableError, match="pip install funasr"):
        engine._load()


def test_funasr_engine_rejects_wrong_sample_rate():
    """16k 是硬要求：传 44.1k 进去不报错的话，识别结果是垃圾而没人知道。"""
    engine = FunAsrEngine()
    with pytest.raises(AudioError, match="16000"):
        engine.transcribe(np.zeros(1000, dtype=np.float32), sample_rate=44100)


def test_segments_from_result_parses_sentence_info():
    from feiyi.rag.audio import _segments_from_result

    item = {
        "text": "整段文本",
        "sentence_info": [
            {"start": 0, "end": 1200, "text": "第一句。", "spk": 0},
            {"start": 1300, "end": 2600, "text": "第二句。", "spk": 1},
            {"start": 2700, "end": 2800, "text": "   "},
        ],
    }
    segs = _segments_from_result(item)
    assert len(segs) == 2, "空白句必须丢掉"
    assert segs[0].speaker == "说话人1"
    assert segs[1].speaker == "说话人2"


# ---------------------------------------------------------------- 波形静音检测
def _tone(n, sr=16000, amp=0.3):
    t = np.arange(n) / sr
    return (amp * np.sin(2 * np.pi * 300 * t)).astype(np.float32)


def test_detect_pauses_finds_silence_between_speech():
    """1 秒声音 + 2 秒静音 + 1 秒声音 → 静音区间应当落在 1~3 秒。"""
    from feiyi.rag.audio import detect_pauses

    sr = 16000
    audio = np.concatenate([_tone(sr), np.zeros(2 * sr, dtype=np.float32), _tone(sr)])
    pauses = detect_pauses(audio, sr)
    assert len(pauses) == 1
    start, end = pauses[0]
    assert 900 <= start <= 1100, f"静音起点应在 1s 附近，实际 {start}ms"
    assert 2900 <= end <= 3100, f"静音终点应在 3s 附近，实际 {end}ms"


def test_detect_pauses_ignores_short_gaps():
    """句内 100ms 换气不该被当成话题切换。"""
    from feiyi.rag.audio import detect_pauses

    sr = 16000
    audio = np.concatenate([_tone(sr), np.zeros(int(0.1 * sr), dtype=np.float32), _tone(sr)])
    assert detect_pauses(audio, sr, min_silence_ms=300) == []


def test_detect_pauses_works_with_noise_floor():
    """现场录音有底噪：用固定阈值会把整段都判成静音，必须用相对噪声底。"""
    from feiyi.rag.audio import detect_pauses

    sr = 16000
    rng = np.random.default_rng(7)
    noise = lambda n: rng.normal(0, 0.01, n).astype(np.float32)  # noqa: E731
    audio = np.concatenate([_tone(sr) + noise(sr), noise(2 * sr), _tone(sr) + noise(sr)])
    pauses = detect_pauses(audio, sr)
    assert len(pauses) == 1
    assert 900 <= pauses[0][0] <= 1200


def test_detect_pauses_empty_audio():
    from feiyi.rag.audio import detect_pauses

    assert detect_pauses(np.zeros(0, dtype=np.float32)) == []


def test_chunks_split_on_waveform_pause_even_if_timestamps_touch():
    """关键回归：标点模型把跨静音的两句合并，时间差是 0，仍必须切开。"""
    tr = _tr(
        [
            Segment(0, 5000, "最后一道是上油，上了油影人才透亮。"),
            Segment(5000, 12000, "唱戏的调子是碗碗腔，乐器简单。"),  # 时间戳首尾相接
        ]
    )
    tr.pauses = [(5000, 6500)]  # 但波形里有 1.5 秒静音

    merged = transcript_chunks(tr, gap_ms=1200, use_pauses=False)
    assert len(merged) == 1, "只看时间差时确实分不开——这正是必须看波形的原因"

    split = transcript_chunks(tr, gap_ms=1200, use_pauses=True)
    assert len(split) == 2
    assert split[1].meta["start_ms"] == 5000


def test_short_waveform_pause_does_not_split():
    tr = _tr([Segment(0, 5000, "前半句。"), Segment(5000, 9000, "后半句。")])
    tr.pauses = [(5000, 5300)]  # 300ms 换气
    assert len(transcript_chunks(tr, gap_ms=1200)) == 1


def test_pause_outside_gap_is_ignored():
    """静音落在段内部（不在两段之间）时不该触发切分。"""
    tr = _tr([Segment(0, 5000, "一段话。"), Segment(5000, 9000, "接着一段话。")])
    tr.pauses = [(9000, 12000)]  # 在两段之后
    assert len(transcript_chunks(tr, gap_ms=1200)) == 1
