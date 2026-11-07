"""编排测试：检索结果必须真的进入 prompt，事件流顺序必须可预测。"""
import asyncio

from feiyi.clients.llm import StubLLM
from feiyi.pipeline import RAGPipeline
from feiyi.rag.chunking import Chunk
from feiyi.rag.embedder import HashingEmbedder
from feiyi.rag.retriever import HybridRetriever

DOCS = [
    Chunk(id="c1", text="选皮多用牛皮或驴皮，制皮时要把皮子浸泡刮薄晾干。", source="皮影制皮"),
    Chunk(id="c2", text="秦腔是陕西地方戏曲，唱腔高亢激越。", source="秦腔简介"),
]


def build_pipeline(llm=None) -> RAGPipeline:
    retriever = HybridRetriever(embedder=HashingEmbedder(), keyword_weight=0.6)
    retriever.index(DOCS)
    return RAGPipeline(retriever=retriever, llm=llm or StubLLM(reply="做皮影先要选牛皮，浸泡刮薄再晾干。"), top_k=1)


def test_build_messages_injects_retrieved_text_and_source():
    pipeline = build_pipeline()
    hits = pipeline.retrieve("牛皮怎么处理")
    messages = pipeline.build_messages("牛皮怎么处理", hits)

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    user_content = messages[1]["content"]
    assert "牛皮" in user_content
    assert "皮影制皮" in user_content
    assert "【观众问题】牛皮怎么处理" in user_content


def test_build_messages_without_hits_states_no_data():
    pipeline = build_pipeline()
    messages = pipeline.build_messages("无关问题", [])
    assert "没有检索到相关资料" in messages[1]["content"]


def test_context_is_truncated_by_budget():
    pipeline = build_pipeline()
    pipeline.max_context_chars = 10
    hits = pipeline.retrieve("牛皮怎么处理")
    messages = pipeline.build_messages("牛皮怎么处理", hits)
    context_part = messages[1]["content"].split("【观众问题】")[0]
    assert len(context_part) < 200


def test_stream_emits_sources_before_tokens():
    pipeline = build_pipeline()

    async def run():
        kinds = []
        async for kind, _payload in pipeline.stream("牛皮怎么处理"):
            kinds.append(kind)
        return kinds

    kinds = asyncio.run(run())
    assert kinds[0] == "sources"
    assert "token" in kinds
    assert kinds.index("sources") < kinds.index("token")


def test_answer_returns_text_and_sources():
    pipeline = build_pipeline()
    answer = asyncio.run(pipeline.answer("牛皮怎么处理"))
    assert "牛皮" in answer.text
    assert answer.sources
    assert answer.sources[0].chunk.source == "皮影制皮"


def test_llm_receives_single_call_per_answer():
    llm = StubLLM(reply="好的。")
    pipeline = build_pipeline(llm=llm)
    asyncio.run(pipeline.answer("牛皮怎么处理"))
    assert len(llm.calls) == 1
    assert "皮影制皮" in llm.calls[0][1]["content"]
