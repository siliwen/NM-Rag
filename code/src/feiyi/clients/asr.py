"""语音识别（ASR）：只负责识别观众的普通话，不处理方言（见 AGENT.md DEC-006）。"""
from __future__ import annotations

from typing import Protocol


class ASRClient(Protocol):
    async def transcribe(self, audio: bytes, sample_rate: int = 16000) -> str: ...


class StubASR:
    """离线桩：返回预设文本，用于串联链路。"""

    def __init__(self, text: str = "皮影戏的牛皮是怎么处理的") -> None:
        self.text = text

    async def transcribe(self, audio: bytes, sample_rate: int = 16000) -> str:
        return self.text


class HttpASR:
    """对接本地 FunASR 服务。约定：POST {base_url}/asr，body 为音频字节，返回 {"text": "..."}。"""

    def __init__(self, base_url: str, timeout: int = 30) -> None:
        if not base_url:
            raise RuntimeError("未配置 ASR 服务地址；测试期请使用 StubASR")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def transcribe(self, audio: bytes, sample_rate: int = 16000) -> str:
        import httpx

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/asr",
                content=audio,
                headers={"Content-Type": "application/octet-stream", "X-Sample-Rate": str(sample_rate)},
            )
            if resp.status_code >= 400:
                raise RuntimeError(f"ASR 服务返回 {resp.status_code}: {resp.text[:200]}")
            return str(resp.json().get("text", ""))
