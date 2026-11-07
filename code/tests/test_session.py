"""会话测试：状态机流转正确，且「说话中被打断」必须真的停下来。"""
import asyncio

from feiyi.clients.llm import StubLLM
from feiyi.clients.tts import StubTTS
from feiyi.pipeline import RAGPipeline
from feiyi.rag.chunking import Chunk
from feiyi.rag.embedder import HashingEmbedder
from feiyi.rag.retriever import HybridRetriever
from feiyi.session import Session, SessionState

DOCS = [Chunk(id="c1", text="选皮多用牛皮或驴皮，制皮时要把皮子浸泡刮薄晾干。", source="皮影制皮")]

LONG_REPLY = "做皮影先要选皮。通常用牛皮或者驴皮。然后把皮子浸泡。再刮薄晾干。皮面要平整透光。这样才好镂刻。"


def build_session(reply: str = LONG_REPLY, delay: float = 0.0) -> Session:
    retriever = HybridRetriever(embedder=HashingEmbedder(), keyword_weight=0.6)
    retriever.index(DOCS)
    pipeline = RAGPipeline(retriever=retriever, llm=StubLLM(reply=reply, delay=delay), top_k=1)
    return Session.create(pipeline=pipeline, tts=StubTTS())


def collect(session: Session, text: str):
    async def run():
        events = []
        async for event in session.say(text):
            events.append(event)
        return events

    return asyncio.run(run())


def test_state_flow_listening_thinking_speaking_idle():
    session = build_session()
    events = collect(session, "牛皮怎么处理")
    states = [e["state"] for e in events if e["type"] == "state"]
    assert states[0] == "listening"
    assert "thinking" in states
    assert "speaking" in states
    assert states[-1] == "idle"


def test_tokens_concatenate_to_done_text():
    session = build_session()
    events = collect(session, "牛皮怎么处理")
    tokens = [e["text"] for e in events if e["type"] == "token"]
    done = [e for e in events if e["type"] == "done"]
    assert done, "正常完成必须产出 done 事件"
    assert "".join(tokens) == done[0]["text"]


def test_sources_are_forwarded_to_client():
    session = build_session()
    events = collect(session, "牛皮怎么处理")
    sources = [e for e in events if e["type"] == "sources"]
    assert sources and sources[0]["items"]
    assert sources[0]["items"][0]["source"] == "皮影制皮"


def test_audio_event_emitted_when_tts_present():
    session = build_session()
    events = collect(session, "牛皮怎么处理")
    audio = [e for e in events if e["type"] == "audio"]
    assert audio and audio[0]["bytes"] > 0


def test_interrupt_stops_generation():
    session = build_session(delay=0.02)

    async def run():
        events = []
        interrupted = False
        async for event in session.say("牛皮怎么处理"):
            events.append(event)
            if not interrupted and sum(1 for e in events if e["type"] == "token") == 1:
                session.interrupt()
                interrupted = True
        return events

    events = asyncio.run(run())
    kinds = [e["type"] for e in events]
    assert "interrupted" in kinds, "打断后必须发出 interrupted 事件"
    assert "done" not in kinds, "被打断的会话不应产出 done"
    assert session.state == SessionState.IDLE
    assert session.interrupt_count == 1


def test_interrupt_before_say_does_not_break_session():
    session = build_session()
    session.interrupt()
    events = collect(session, "牛皮怎么处理")
    assert any(e["type"] == "done" for e in events)


def test_history_is_recorded():
    session = build_session()
    collect(session, "牛皮怎么处理")
    assert len(session.history) == 2
    assert session.history[0]["role"] == "user"
    assert session.history[1]["role"] == "assistant"


def test_history_respects_limit():
    session = build_session(reply="短答。")
    session.history_limit = 2
    collect(session, "问题一")
    collect(session, "问题二")
    assert len(session.history) == 2
