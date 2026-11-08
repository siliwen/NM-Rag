"""会话状态机：管理一次对话的生命周期，支持「说话中被打断」。

TTS 按句流式：大模型吐完一句就送去合成，合成完立刻下发一段音频，
不必等全文生成完再整段合成 —— 首音延迟从「全文合成」降到「第一句合成」。

实现方式：两个后台协程（文本生产 / 音频合成）各自往同一个事件队列里塞事件，
say() 只负责按到达顺序往外 yield，因此 token 和 audio 可以交错到达。
音频的合成是串行的（按句序），既保证顺序又避免多路抢占 CPU。

状态流转：
    idle → listening → thinking → speaking → idle
    任意状态收到 interrupt() → idle（丢弃尚未产出的 token 与未播完的音频）
"""
from __future__ import annotations

import asyncio
import base64
import time
import uuid
from enum import Enum
from typing import Any, AsyncIterator

from .clients.avatar import AvatarClient
from .clients.tts import TTSClient
from .pipeline import RAGPipeline
from .textsplit import pop_complete_sentences

# 句子的最小合成长度：太短的句子会合成出一截一截的音频，拼起来发硬
MIN_TTS_CHARS = 4
# 缓冲区超过这个长度仍无句末标点就硬切，避免数字人长时间哑着不出声
MAX_TTS_CHARS = 40
# 第一段专门用更小的窗口：Kokoro 实测每字约 50ms，23 字要合成 1.6 秒，
# 压到 12 字只要 0.85 秒。首段先开口，后面播着的时候再按正常粒度合成，
# 听众感知不到后面的切分点。
FIRST_MIN_TTS_CHARS = 4
FIRST_MAX_TTS_CHARS = 12


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
        """立刻打断：递增代次号，正在进行的 say() 会在下一个事件处退出。"""
        self._generation += 1
        self.interrupt_count += 1
        self.state = SessionState.IDLE
        return True

    def _push_history(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.history_limit:
            self.history = self.history[-self.history_limit :]

    async def _produce(
        self,
        text: str,
        generation: int,
        events: asyncio.Queue,
        sentences: asyncio.Queue,
        collected: list[str],
        speak: bool,
        plan: Any = None,
    ) -> None:
        """读大模型流：token 直接进事件队列，攒够一句就丢进合成队列。"""
        buffer = ""
        first_spoken = False
        try:
            async for kind, payload in self.pipeline.stream(text, plan):
                if generation != self._generation:
                    return
                if kind == "sources":
                    await events.put(("sources", payload))
                elif kind == "token":
                    token = str(payload)
                    collected.append(token)
                    await events.put(("token", token))
                    if speak:
                        buffer += token
                        ready, buffer = pop_complete_sentences(
                            buffer,
                            min_chars=FIRST_MIN_TTS_CHARS if not first_spoken else MIN_TTS_CHARS,
                            max_chars=FIRST_MAX_TTS_CHARS if not first_spoken else MAX_TTS_CHARS,
                            # 首段允许在逗号处就切：等句号可能要白等半秒以上
                            allow_break=not first_spoken,
                        )
                        for sentence in ready:
                            await sentences.put(sentence)
                            first_spoken = True
            # 流结束：剩下的半句也要合成，否则最后一句话永远没声音
            if speak and generation == self._generation:
                tail, _ = pop_complete_sentences(
                    buffer, min_chars=MIN_TTS_CHARS, max_chars=MAX_TTS_CHARS, flush=True
                )
                for sentence in tail:
                    await sentences.put(sentence)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # NO-006：生成失败必须显式上抛，不静默
            await events.put(("error", f"生成失败：{exc}"))
        finally:
            await events.put(("text_done", None))
            if speak:
                await sentences.put(None)

    async def _speak(self, generation: int, events: asyncio.Queue, sentences: asyncio.Queue) -> None:
        """按句序合成并下发音频；单句失败只上报，不中断整轮回答。"""
        seq = 0
        try:
            while True:
                sentence = await sentences.get()
                if sentence is None:
                    return
                if generation != self._generation:
                    return
                try:
                    started = time.perf_counter()
                    audio = await self.tts.synthesize(sentence)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    await events.put(("error", f"语音合成失败：{exc}"))
                    continue
                if generation != self._generation:
                    return
                seq += 1
                await events.put(
                    (
                        "audio",
                        {
                            "type": "audio",
                            "seq": seq,
                            "text": sentence,
                            "elapsed_ms": round((time.perf_counter() - started) * 1000),
                            "bytes": len(audio),
                            "mime_type": "audio/wav",
                            "data": base64.b64encode(audio).decode("ascii"),
                        },
                    )
                )
        finally:
            await events.put(("audio_done", None))

    async def say(self, text: str) -> AsyncIterator[dict]:
        """以事件流的形式完成一次回答。事件可直接透传给前端（NDJSON / SSE）。"""
        self._generation += 1
        generation = self._generation

        self.state = SessionState.LISTENING
        yield {"type": "state", "state": self.state.value}

        self.state = SessionState.THINKING
        yield {"type": "state", "state": self.state.value}

        events: asyncio.Queue = asyncio.Queue()
        sentences: asyncio.Queue = asyncio.Queue()
        collected: list[str] = []
        sources: list[dict] = []

        # 清单型问题（「列出所有…」）不合成语音：回答上千字、上百句，
        # 逐句合成要好几分钟，而且用听的方式列清单本身没有意义。
        # 判断交给 pipeline：像列举句式但只命中一张卡（「秦腔有哪些角色行当」）
        # 仍按普通问答走，语音必须保留。
        plan = self.pipeline.plan(text)
        speak = self.tts is not None and plan.mode != "enumerate"
        producer = asyncio.create_task(
            self._produce(text, generation, events, sentences, collected, speak, plan)
        )
        consumer = (
            asyncio.create_task(self._speak(generation, events, sentences)) if speak else None
        )

        text_done = False
        audio_done = not speak
        speaking = False

        try:
            while not (text_done and audio_done):
                kind, payload = await events.get()

                if generation != self._generation:
                    yield {"type": "interrupted"}
                    yield {"type": "state", "state": SessionState.IDLE.value}
                    return

                if kind == "token":
                    if not speaking:
                        speaking = True
                        self.state = SessionState.SPEAKING
                        yield {"type": "state", "state": self.state.value}
                    yield {"type": "token", "text": payload}
                elif kind == "sources":
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
                elif kind == "audio":
                    yield payload
                elif kind == "error":
                    yield {"type": "error", "state": payload}
                elif kind == "text_done":
                    text_done = True
                elif kind == "audio_done":
                    audio_done = True
        finally:
            producer.cancel()
            pending = [producer]
            if consumer is not None:
                consumer.cancel()
                pending.append(consumer)
            await asyncio.gather(*pending, return_exceptions=True)

        full_text = "".join(collected)

        if self.avatar is not None and full_text.strip():
            await self.avatar.say(full_text, self.id)

        self._push_history("user", text)
        self._push_history("assistant", full_text)
        self.state = SessionState.IDLE
        yield {"type": "done", "text": full_text}
        yield {"type": "state", "state": self.state.value}
