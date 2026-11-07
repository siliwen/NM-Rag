"""语音合成（TTS）。

生产实现必须是本地推理（CosyVoice2 / GPT-SoVITS），
禁止使用 EdgeTTS 等联网且 GPL 协议的方案（见 AGENT.md NO-009 / DEC-011）。
"""
from __future__ import annotations

import io
import wave
from typing import Protocol


class TTSClient(Protocol):
    async def synthesize(self, text: str) -> bytes: ...


def silence_wav(seconds: float, sample_rate: int = 16000) -> bytes:
    """生成指定时长的静音 wav，用于占位与端到端联调。"""
    frames = max(1, int(seconds * sample_rate))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * frames)
    return buf.getvalue()


class StubTTS:
    """离线桩：按字数估算时长生成静音 wav，不联网、不占显存。"""

    def __init__(self, seconds_per_char: float = 0.18, sample_rate: int = 16000) -> None:
        self.seconds_per_char = seconds_per_char
        self.sample_rate = sample_rate

    async def synthesize(self, text: str) -> bytes:
        duration = max(0.2, len(text or "") * self.seconds_per_char)
        return silence_wav(duration, self.sample_rate)


class HttpTTS:
    """对接本地 TTS 服务（CosyVoice2 / GPT-SoVITS 自行封装 HTTP 接口）。

    约定：POST {base_url}/tts，body={"text": "..."}，返回 audio/wav 字节流。
    """

    def __init__(self, base_url: str, timeout: int = 30, voice_id: str = "") -> None:
        if not base_url:
            raise RuntimeError("未配置 TTS 服务地址；测试期请使用 StubTTS")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.voice_id = voice_id

    async def synthesize(self, text: str) -> bytes:
        import httpx

        payload = {"text": text}
        if self.voice_id:
            payload["voice_id"] = self.voice_id
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.base_url}/tts", json=payload)
            if resp.status_code >= 400:
                raise RuntimeError(f"TTS 服务返回 {resp.status_code}: {resp.text[:200]}")
            return resp.content
