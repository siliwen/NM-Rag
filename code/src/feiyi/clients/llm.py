"""大模型客户端。唯一允许联网与产生费用的环节。

生产统一按 OpenAI 兼容协议对接：阿里云百炼 / 腾讯云 / 学校自建云，只换 base_url。
"""
from __future__ import annotations

import json
import re
from typing import AsyncIterator, Protocol, Sequence

Message = dict


class LLMClient(Protocol):
    async def complete(self, messages: Sequence[Message], temperature: float = 0.3) -> str: ...

    def stream(self, messages: Sequence[Message], temperature: float = 0.3) -> AsyncIterator[str]: ...


class StubLLM:
    """离线桩：不联网、可复现。用于测试与无密钥时的端到端联调。"""

    def __init__(self, reply: str | None = None, delay: float = 0.0) -> None:
        self.reply = reply
        self.delay = delay
        self.calls: list[list[Message]] = []

    def _render(self, messages: Sequence[Message]) -> str:
        if self.reply is not None:
            return self.reply
        last_user = ""
        for m in reversed(list(messages)):
            if m.get("role") == "user":
                last_user = str(m.get("content", ""))
                break
        question = last_user.split("【观众问题】")[-1].strip() or last_user
        return f"这是关于「{question[:20]}」的讲解。"

    async def complete(self, messages: Sequence[Message], temperature: float = 0.3) -> str:
        self.calls.append(list(messages))
        return self._render(messages)

    async def stream(self, messages: Sequence[Message], temperature: float = 0.3) -> AsyncIterator[str]:
        import asyncio

        self.calls.append(list(messages))
        text = self._render(messages)
        for piece in re.findall(r"[^，。！？；]*[，。！？；]?", text):
            if not piece:
                continue
            if self.delay:
                await asyncio.sleep(self.delay)
            yield piece


class OpenAICompatLLM:
    """OpenAI 兼容接口（/chat/completions）。流式与非流式都支持。"""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: int = 60,
        max_tokens: int = 800,
    ) -> None:
        if not api_key:
            raise RuntimeError("缺少 LLM_API_KEY，请在 .env 中配置，或使用 StubLLM 做离线联调")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens

    @property
    def _endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _payload(self, messages: Sequence[Message], temperature: float, stream: bool) -> dict:
        return {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }

    async def complete(self, messages: Sequence[Message], temperature: float = 0.3) -> str:
        import httpx

        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                self._endpoint, headers=headers, json=self._payload(messages, temperature, False)
            )
            if resp.status_code >= 400:
                raise RuntimeError(f"LLM 请求失败 {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
        return data["choices"][0]["message"]["content"]

    async def stream(self, messages: Sequence[Message], temperature: float = 0.3) -> AsyncIterator[str]:
        import httpx

        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST", self._endpoint, headers=headers, json=self._payload(messages, temperature, True)
            ) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    raise RuntimeError(f"LLM 流式请求失败 {resp.status_code}: {body[:300]}")
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    chunk = line[len("data:") :].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        delta = json.loads(chunk)["choices"][0].get("delta", {})
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
                    piece = delta.get("content")
                    if piece:
                        yield piece


def build_llm(settings) -> LLMClient:
    """按配置构造 LLM 客户端；未配置密钥时退回可离线运行的 StubLLM。"""
    if not settings.llm_configured:
        return StubLLM()
    return OpenAICompatLLM(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        timeout=settings.llm_timeout,
    )
