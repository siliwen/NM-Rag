"""语音合成（TTS）。

生产实现必须是本地推理（CosyVoice2 / GPT-SoVITS），
禁止使用 EdgeTTS 等联网且 GPL 协议的方案（见 AGENT.md NO-009 / DEC-011）。
"""
from __future__ import annotations

import io
import wave
from typing import Any, Protocol


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
    """对接本地 TTS 服务（Kokoro / CosyVoice2 / GPT-SoVITS 自行封装 HTTP 接口）。

    约定：POST {base_url}/tts，body={"text": "...", "voice_id": "..."}，返回 audio/wav 字节流。

    `trust_env=False`：本地服务不能走系统代理。本机实测 HTTP_PROXY 指向本地代理时，
    每次调用会被中转多花约 0.6 秒（合成本身才 1.3 秒），首音延迟直接翻倍。
    """

    def __init__(
        self, base_url: str, timeout: int = 30, voice_id: str = "", trust_env: bool = False
    ) -> None:
        if not base_url:
            raise RuntimeError("未配置 TTS 服务地址；测试期请使用 StubTTS")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.voice_id = voice_id
        self.trust_env = trust_env
        # 连接池：一段回答要合成 4~6 次，每次重建客户端等于多付 4~6 次 TCP 握手
        # （本机实测每段约 0.29 秒，首段延迟直接多出三成）
        self._client: Any = None

    def _connect(self) -> Any:
        import httpx

        client = self._client
        if client is None or getattr(client, "is_closed", False):
            client = httpx.AsyncClient(timeout=self.timeout, trust_env=self.trust_env)
            self._client = client
        return client

    async def synthesize(self, text: str) -> bytes:
        client = self._connect()
        payload = {"text": text}
        if self.voice_id:
            payload["voice_id"] = self.voice_id
        resp = await client.post(f"{self.base_url}/tts", json=payload)
        if resp.status_code >= 400:
            raise RuntimeError(f"TTS 服务返回 {resp.status_code}: {resp.text[:200]}")
        return resp.content

    async def aclose(self) -> None:
        client, self._client = self._client, None
        if client is not None and not getattr(client, "is_closed", False):
            await client.aclose()

    async def warmup(self, text: str = "好。") -> None:
        """启动预热：建好连接池 + 让后端把模型加载进内存。

        不做这一步，第一次提问要额外付 TCP 握手（~0.29s）和后端冷加载（十几秒）。
        失败必须静默：TTS 后端可能比编排层晚启动，不能因此让服务起不来。
        """
        try:
            await self.synthesize(text)
        except Exception as exc:  # 预热失败只提示，不阻断启动
            print(f"[TTS] 预热未成功（不影响启动，首次合成会慢一些）：{exc}")
