"""渲染引擎客户端：对接本机 LiveTalking 的 HTTP 接口。

LiveTalking 关键接口（参考 docs/api.md）：
- POST /human        {"text": "...", "sessionid": "...", "interrupt": true|false}
- POST /humanaudio   直接推一段音频
"""
from __future__ import annotations

from typing import Protocol


class AvatarClient(Protocol):
    async def say(self, text: str, session_id: str) -> dict: ...

    async def interrupt(self, session_id: str) -> dict: ...


class LiveTalkingClient:
    def __init__(self, base_url: str, avatar_id: str = "", timeout: int = 15) -> None:
        self.base_url = base_url.rstrip("/")
        self.avatar_id = avatar_id
        self.timeout = timeout

    async def say(self, text: str, session_id: str) -> dict:
        import httpx

        payload = {"text": text, "sessionid": session_id, "interrupt": True, "type": "echo"}
        if self.avatar_id:
            payload["avatar_id"] = self.avatar_id
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.base_url}/human", json=payload)
            if resp.status_code >= 400:
                raise RuntimeError(f"渲染引擎返回 {resp.status_code}: {resp.text[:200]}")
            return {"ok": True, "status": resp.status_code}

    async def interrupt(self, session_id: str) -> dict:
        import httpx

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/human", json={"text": "", "sessionid": session_id, "interrupt": True}
            )
            if resp.status_code >= 400:
                raise RuntimeError(f"打断失败 {resp.status_code}: {resp.text[:200]}")
            return {"ok": True, "status": resp.status_code}
