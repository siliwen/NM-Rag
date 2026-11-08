"""Minimal local HTTP adapter for Kokoro Chinese TTS (CPU-compatible).

权重优先级：环境变量 KOKORO_MODEL_DIR 指向的本地目录 > HuggingFace（repo_id）。
本机实测 huggingface 镜像会在下载中途停滞，所以默认先看 data/models/kokoro-82m-zh
（由 scripts/fetch_kokoro_model.py 从 ModelScope 下好）。

性能取舍：整段一次性合成时，首音要等「全文合成完」，纯等待 ≈ 音频时长 × RTF。
所以这里按句切分后**并行**合成再拼接，RTF 从 0.30 降到约 0.10，
代价是多线程各自算一小段（结果按原顺序拼接，听感无差别）。

切句逻辑与编排层共用 `feiyi.textsplit`：否则「按句流式」和「分段并行」两处口径
会各自演化，同一句话在两边切得不一样。
"""
from __future__ import annotations

import io
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from feiyi.textsplit import split_sentences  # noqa: E402

MODEL_ID = os.getenv("KOKORO_MODEL_ID", "hexgrad/Kokoro-82M-v1.1-zh")
MODEL_DIR = Path(os.getenv("KOKORO_MODEL_DIR", str(ROOT / "data" / "models" / "kokoro-82m-zh")))
DEFAULT_VOICE = os.getenv("KOKORO_VOICE_ID", "zf_001")
SAMPLE_RATE = 24000  # Kokoro 固定 24k，别跟着默认 16k 写字面量

# 语速：1.0 偏慢（实测 4.9 字/秒 + 32% 静音），1.15 约为正常播报速度
SPEED = float(os.getenv("KOKORO_SPEED", "1.15"))
# 并行度：4 段同时合成；CPU 核数少时自动降到分片数
WORKERS = int(os.getenv("KOKORO_WORKERS", "4"))
# 单个分片的最大字数：太长则并行粒度粗，太短则拼接处容易生硬
MAX_CHARS = int(os.getenv("KOKORO_MAX_CHARS", "40"))
# 拼接过短的分句会导致频繁断句，低于这个字数就合并到下一句
MIN_CHARS = int(os.getenv("KOKORO_MIN_CHARS", "4"))
# 分片之间额外补的静音（毫秒）。Kokoro 自己会在句末留尾音，默认不再加
GAP_MS = int(os.getenv("KOKORO_GAP_MS", "0"))
# 单次停顿的上限：超过就削到这里，实测长停顿 0.8~1.1s 是最“拖”的部分
MAX_PAUSE_MS = int(os.getenv("KOKORO_MAX_PAUSE_MS", "420"))
# 判定静音的 RMS 阈值（float 幅度）
SILENCE_THRESHOLD = float(os.getenv("KOKORO_SILENCE_RMS", "0.01"))

app = FastAPI(title="Local Kokoro TTS")


class SynthesisRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    voice_id: str | None = None
    speed: float | None = Field(default=None, gt=0.3, lt=2.0)


def local_weights() -> tuple[Path, Path] | None:
    """本地权重齐全时返回 (config, model)；缺一个就当没下完，避免半套文件报奇怪的错。"""
    config = MODEL_DIR / "config.json"
    model = MODEL_DIR / "kokoro-v1_1-zh.pth"
    return (config, model) if config.exists() and model.exists() else None


@lru_cache(maxsize=1)
def get_pipeline():
    from kokoro import KPipeline

    weights = local_weights()
    if weights is not None:
        from kokoro.model import KModel

        config, model = weights
        kmodel = KModel(repo_id=MODEL_ID, config=str(config), model=str(model))
        return KPipeline(lang_code="z", repo_id=MODEL_ID, model=kmodel)
    return KPipeline(lang_code="z", repo_id=MODEL_ID)


def resolve_voice(voice_id: str) -> str:
    """本地有音色包就走本地路径（KPipeline 认 .pt 绝对路径），否则交给它去 HF 拉。"""
    local = MODEL_DIR / "voices" / f"{voice_id}.pt"
    return str(local) if local.exists() else voice_id


def _synth_one(pipeline, text: str, voice: str, speed: float) -> np.ndarray:
    chunks = [audio for _, _, audio in pipeline(text, voice=voice, speed=speed)]
    if not chunks:
        raise RuntimeError(f"Kokoro 没有生成音频：{text[:20]}")
    return np.concatenate(chunks)


def _join(chunks: list[np.ndarray]) -> np.ndarray:
    if GAP_MS <= 0:
        return np.concatenate(chunks)
    gap = np.zeros(int(SAMPLE_RATE * GAP_MS / 1000), dtype=np.float32)
    joined: list[np.ndarray] = []
    for idx, chunk in enumerate(chunks):
        if idx:
            joined.append(gap)
        joined.append(chunk)
    return np.concatenate(joined)


def cap_silence(audio: np.ndarray, max_pause_ms: int = MAX_PAUSE_MS) -> np.ndarray:
    """把超过上限的停顿削短，短停顿原样保留。

    实测 Kokoro 在句号处会留 0.8~1.1 秒的静音，一句一段地听到最后像是"卡住了"；
    削到 0.4 秒左右既保留语气，又不至于让整段拖长。切点落在静音里，不会爆音。
    """
    if max_pause_ms <= 0 or audio.size == 0:
        return audio
    frame = max(1, int(SAMPLE_RATE * 0.02))
    n_frames = audio.size // frame
    if n_frames == 0:
        return audio
    env = np.sqrt(np.mean(audio[: n_frames * frame].reshape(n_frames, frame) ** 2, axis=1))
    quiet = env < SILENCE_THRESHOLD

    max_frames = max(1, int(max_pause_ms / 20))
    keep: list[np.ndarray] = []
    start = 0
    idx = 0
    while idx < n_frames:
        if not quiet[idx]:
            idx += 1
            continue
        run_start = idx
        while idx < n_frames and quiet[idx]:
            idx += 1
        run_len = idx - run_start
        if run_len <= max_frames:
            continue  # 正常语气停顿，不动
        # 保留这段停顿的前 max_frames，其余丢掉；切点仍在静音里，不会爆音
        keep.append(audio[start:(run_start + max_frames) * frame])
        start = idx * frame
    keep.append(audio[start:])
    return np.concatenate(keep) if keep else audio



def synth_all(
    pipeline,
    text: str,
    voice: str,
    speed: float = SPEED,
    workers: int = WORKERS,
) -> tuple[np.ndarray, list[str]]:
    """切句 → 并行合成 → 按原顺序拼接。返回 (音频, 分片文本)。"""
    parts = split_sentences(text, max_chars=MAX_CHARS, min_chars=MIN_CHARS)
    if not parts:
        raise ValueError("文本为空，无法合成")
    if len(parts) == 1:
        return _synth_one(pipeline, parts[0], voice, speed), parts
    if workers <= 1:
        # 串行也要把每一片走完：曾经这里只取 parts[0]，其余分片被静默丢掉
        return _join([_synth_one(pipeline, p, voice, speed) for p in parts]), parts

    # 先把音色包灌进缓存，否则每个线程都会各自 torch.load 一遍
    loader = getattr(pipeline, "load_voice", None)
    if callable(loader):
        loader(voice)

    with ThreadPoolExecutor(max_workers=min(workers, len(parts))) as pool:
        chunks = list(pool.map(lambda p: _synth_one(pipeline, p, voice, speed), parts))

    return _join(chunks), parts


@app.get("/health")
def health() -> dict[str, str]:
    weights = local_weights()
    return {
        "status": "ok",
        "model": str(MODEL_DIR) if weights else MODEL_ID,
        "model_source": "local" if weights else "huggingface",
        "voice_id": DEFAULT_VOICE,
        "speed": str(SPEED),
        "workers": str(WORKERS),
        "max_pause_ms": str(MAX_PAUSE_MS),
    }


@app.post("/tts")
def synthesize(body: SynthesisRequest) -> Response:
    try:
        pipeline = get_pipeline()
        voice = resolve_voice(body.voice_id or DEFAULT_VOICE)
        speed = body.speed or SPEED
        audio, _ = synth_all(pipeline, body.text, voice=voice, speed=speed)
        audio = cap_silence(audio)
        wav = io.BytesIO()
        sf.write(wav, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        return Response(wav.getvalue(), media_type="audio/wav")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"本地 TTS 合成失败：{exc}") from exc
