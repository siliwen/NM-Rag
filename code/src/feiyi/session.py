"""会话状态机：管理一次对话的生命周期，支持「说话中被打断」。

状态流转：
    idle → listening → thinking → speaking → idle
    任意状态收到 interrupt() → idle（并丢弃尚未产出的 token）
"""
from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, AsyncIterator

from .clients.avatar import AvatarClient
from .clients.tts import TTSClient
from .pipeline import RAGPipeline


class SessionState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


class Session:
    def __init__(
        self,
        session_id: str,
        pipeline: RAGPipeline,
        tts: TTSClient | None = None,
        avatar: AvatarClient | None = None,
        history_limit: int = 20,
    ) -> None:
        self.id = session_id
        self.pipeline = pipeline
        self.tts = tts
        self.avatar = avatar
        self.history_limit = history_limit
        self.state = SessionState.IDLE
        self.history: list[dict] = []
        self._generation = 0
        self.interrupt_count = 0

    @classmethod
    def create(cls, pipeline: RAGPipeline, **kwargs: Any) -> "Session":
        return cls(session_id=uuid.uuid4().hex[:12], pipeline=pipeline, **kwargs)

    def interrupt(self) -> bool:
        """立刻打断：递增代次号，正在进行的 say() 会在下一个 token 处退出。"""
        self._generation += 1
        self.interrupt_count += 1
        self.state = SessionState.IDLE
        return True

    def _push_history(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.history_limit:
            self.history = self.history[-self.history_limit :]

    async def say(self, text: str) -> AsyncIterator[dict]:
        """以事件流的形式完成一次回答。事件可直接透传给前端（NDJSON / SSE）。"""
        self._generation += 1
        generation = self._generation

        self.state = SessionState.LISTENING
        yield {"type": "state", "state": self.state.value}

        self.state = SessionState.THINKING
        yield {"type": "state", "state": self.state.value}

        collected: list[str] = []
        sources: list[dict] = []
        speaking = False

        async for kind, payload in self.pipeline.stream(text):
            if generation != self._generation:
                yield {"type": "interrupted"}
                yield {"type": "state", "state": SessionState.IDLE.value}
                return

            if kind == "sources":
                sources = [
                    {
                        "id": hit.chunk.id,
                        "source": hit.chunk.source,
                        "page": hit.chunk.meta.get("page"),
                        "ocr": hit.chunk.meta.get("ocr"),
                        "start_ms": hit.chunk.meta.get("start_ms"),
                        "end_ms": hit.chunk.meta.get("end_ms"),
                        "score": round(float(hit.score), 4),
                        "text": hit.chunk.text,
                        "detail": hit.detail,
                    }
                    for hit in payload
                ]
                yield {"type": "sources", "items": sources}
            elif kind == "token":
                if not speaking:
                    speaking = True
                    self.state = SessionState.SPEAKING
                    yield {"type": "state", "state": self.state.value}
                collected.append(str(payload))
                yield {"type": "token", "text": str(payload)}

        full_text = "".join(collected)

        if generation != self._generation:
            yield {"type": "interrupted"}
            yield {"type": "state", "state": SessionState.IDLE.value}
            return

        if self.tts is not None and full_text.strip():
            audio = await self.tts.synthesize(full_text)
            if generation == self._generation:
                yield {"type": "audio", "bytes": len(audio)}

        if self.avatar is not None and full_text.strip():
            await self.avatar.say(full_text, self.id)

        self._push_history("user", text)
        self._push_history("assistant", full_text)
        self.state = SessionState.IDLE
        yield {"type": "done", "text": full_text}
        yield {"type": "state", "state": self.state.value}
