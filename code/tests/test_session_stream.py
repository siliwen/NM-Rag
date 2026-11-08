"""按句流式 TTS：一句一合成、按句序下发，失败必须显式上报。

这些用例保护三个容易退化的行为：
1. 音频按句子切分（不是整段一坨），首音要在全文生成完之前就到
2. 被打断后不能继续往下播
3. 合成失败要出 error 事件，不能静默跳过
"""
import asyncio

from feiyi.clients.llm import StubLLM
from feiyi.pipeline import RAGPipeline
from feiyi.rag.chunking import Chunk
from feiyi.rag.embedder import HashingEmbedder
from feiyi.rag.retriever import HybridRetriever
from feiyi.session import Session, SessionState

DOCS = [Chunk(id="c1", text="选皮多用牛皮或驴皮，制皮时浸泡刮薄晾干。", source="皮影制皮")]
REPLY = "先要选皮。通常用牛皮或者驴皮。然后把皮子浸泡刮薄。"


class RecordingTTS:
    """记录每次合成的文本，可选延迟/指定句子失败。"""

    def __init__(self, delay: float = 0.0, fail_on: str = "") -> None:
        self.delay = delay
        self.fail_on = fail_on
        self.calls: list[str] = []

    async def synthesize(self, text: str) -> bytes:
        self.calls.append(text)
        if self.fail_on and self.fail_on in text:
            raise RuntimeError("模拟合成失败")
        if self.delay:
            await asyncio.sleep(self.delay)
        return b"WAV" + text.encode("utf-8")


_UNSET = object()


def build(reply: str = REPLY, delay: float = 0.0, tts=_UNSET) -> Session:
    retriever = HybridRetriever(embedder=HashingEmbedder(), keyword_weight=0.6)
    retriever.index(DOCS)
    pipeline = RAGPipeline(retriever=retriever, llm=StubLLM(reply=reply, delay=delay), top_k=1)
    client = RecordingTTS() if tts is _UNSET else tts
    return Session.create(pipeline=pipeline, tts=client)


def collect(session: Session, text: str = "牛皮怎么处理") -> list[dict]:
    async def run():
        return [event async for event in session.say(text)]

    return asyncio.run(run())


def test_audio_is_split_by_sentence_and_keeps_order():
    tts = RecordingTTS()
    events = collect(build(tts=tts))
    audio = [e for e in events if e["type"] == "audio"]

    assert len(audio) >= 2, "多句回答必须拆成多段音频，否则等于没做流式"
    assert [a["seq"] for a in audio] == list(range(1, len(audio) + 1))
    assert [a["text"] for a in audio] == tts.calls
    assert "".join(a["text"] for a in audio) == REPLY
    assert all(a["data"] and a["mime_type"] == "audio/wav" for a in audio)


def test_first_audio_arrives_before_text_finishes():
    """核心指标：首段音频必须早于最后一个 token，否则首音延迟没改善。"""
    events = collect(build(delay=0.01))
    kinds = [e["type"] for e in events]
    first_audio = kinds.index("audio")
    last_token = len(kinds) - 1 - kinds[::-1].index("token")
    assert first_audio < last_token


def test_first_audio_uses_short_chunk_to_start_early():
    """首段每多一个字，合成多约 50ms（实测 Kokoro）；首段必须压进很短的窗口。

    实测：23 字首段合成 1.63 秒，12 字约 0.85 秒 —— 差的就是用户感觉到的「晚开口」。
    """
    long_first = "皮影戏就是用灯光把皮做的影人照在幕布上讲故事，也叫灯影戏。还有第二句在这里。"
    tts = RecordingTTS()
    events = collect(build(reply=long_first, tts=tts))
    audio = [e for e in events if e["type"] == "audio"]

    assert len(audio[0]["text"]) <= 12, f"首段太长：{audio[0]['text']}"
    assert "".join(a["text"] for a in audio) == long_first, "切短不能丢字"


def test_tts_failure_emits_error_event_and_continues():
    tts = RecordingTTS(fail_on="选皮")
    events = collect(build(tts=tts))
    kinds = [e["type"] for e in events]

    assert "error" in kinds, "合成失败必须显式上报（NO-006），不能静默跳过"
    assert "done" in kinds, "单句合成失败不应中断整轮回答"
    assert len(tts.calls) >= 2, "失败的句子之后仍应继续合成剩余句子"


def test_interrupt_stops_audio_midway():
    session = build(delay=0.01)

    async def run():
        events, stopped = [], False
        async for event in session.say("牛皮怎么处理"):
            events.append(event)
            if not stopped and event["type"] == "audio":
                session.interrupt()
                stopped = True
        return events

    events = asyncio.run(run())
    kinds = [e["type"] for e in events]
    assert "interrupted" in kinds
    assert "done" not in kinds
    assert session.state == SessionState.IDLE


def test_no_tts_means_no_audio_events_but_answer_completes():
    events = collect(build(tts=None))
    assert not [e for e in events if e["type"] == "audio"]
    assert any(e["type"] == "done" for e in events)


def test_empty_reply_does_not_hang():
    events = collect(build(reply="   "))
    assert any(e["type"] == "done" for e in events)
    assert not [e for e in events if e["type"] == "audio"]


def build_resume(reply: str = REPLY, tts=_UNSET, top_k: int = 1) -> Session:
    """简历这类「一个来源很多张卡」的语料，才能触发清单模式。"""
    resume = [
        Chunk(id=f"r{i}", text=f"项目{i}：第{i}个项目的简介与成果。", source="简历") for i in range(8)
    ]
    retriever = HybridRetriever(embedder=HashingEmbedder(), keyword_weight=0.6)
    retriever.index(resume)
    pipeline = RAGPipeline(retriever=retriever, llm=StubLLM(reply=reply), top_k=top_k)
    client = RecordingTTS() if tts is _UNSET else tts
    return Session.create(pipeline=pipeline, tts=client)


def test_enumeration_question_skips_tts():
    """清单型回答动辄上千字、上百句，逐句合成要好几分钟；而且「听清单」本身没有意义。

    实测：46 张卡的项目清单回答 2000+ 字，串行 TTS 把总耗时从 80s 拉到 5 分钟以上。
    """
    tts = RecordingTTS()
    events = collect(build_resume(tts=tts), text="把简历里所有项目列一个清单出来")

    assert tts.calls == [], f"清单型问题不应触发合成，实际合成了 {len(tts.calls)} 句"
    assert not [e for e in events if e["type"] == "audio"]
    assert any(e["type"] == "done" for e in events), "不合成也要正常收尾"


def test_list_style_question_with_one_card_still_speaks():
    """「秦腔有哪些角色行当」是列举句式，但只命中一张卡，不能被当成清单静音。

    改前按问题文本判断，这类问题会突然没声音——用户看起来像是坏了。
    """
    tts = RecordingTTS()
    events = collect(build(tts=tts, reply="秦腔分四生六旦二净一丑。"), text="秦腔有哪些角色行当？")

    audio = [e for e in events if e["type"] == "audio"]
    assert tts.calls, "单卡来源的列举问题必须保留语音"
    assert audio
    assert any(e["type"] == "done" for e in events)

