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
    assert "【用户问题】牛皮怎么处理" in user_content


def test_build_messages_without_hits_states_no_data():
    pipeline = build_pipeline()
    messages = pipeline.build_messages("无关问题", [])
    assert "没有检索到相关资料" in messages[1]["content"]


def test_context_is_truncated_by_budget():
    pipeline = build_pipeline()
    pipeline.max_context_chars = 10
    hits = pipeline.retrieve("牛皮怎么处理")
    messages = pipeline.build_messages("牛皮怎么处理", hits)
    context_part = messages[1]["content"].split("【用户问题】")[0]
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


# ---------- 清单型问题（「把所有项目列出来」这类，必须取全量而不是 top_k）----------
RESUME = [
    Chunk(id=f"r{i}", text=f"项目{i}：第{i}个项目的描述内容，包含简介与成果。", source="简历")
    for i in range(12)
]


class RecordingLLM:
    """记录收到的 max_tokens，用于验证「清单模式要放宽输出上限」。"""

    def __init__(self, reply: str = "ok") -> None:
        self.reply = reply
        self.calls: list[list[dict]] = []
        self.max_tokens_seen: list[int | None] = []

    async def complete(self, messages, temperature: float = 0.3, max_tokens: int | None = None) -> str:
        self.calls.append(list(messages))
        self.max_tokens_seen.append(max_tokens)
        return self.reply

    async def stream(self, messages, temperature: float = 0.3, max_tokens: int | None = None):
        self.calls.append(list(messages))
        self.max_tokens_seen.append(max_tokens)
        yield self.reply


def build_big(llm=None, top_k: int = 3) -> RAGPipeline:
    retriever = HybridRetriever(embedder=HashingEmbedder(), keyword_weight=0.6)
    retriever.index(DOCS + RESUME)
    return RAGPipeline(retriever=retriever, llm=llm or RecordingLLM(), top_k=top_k)


def test_is_enumeration_detects_list_style_questions():
    from feiyi.pipeline import is_enumeration

    assert is_enumeration("把简历里所有项目列一个清单出来")
    assert is_enumeration("宋伟做过哪些项目")
    assert is_enumeration("请汇总一下全部成果")
    # 普通讲解问题不能被误判，否则每次问答都塞满上下文
    assert not is_enumeration("皮影戏是怎么做的")
    assert not is_enumeration("秦腔的唱腔有什么特点")


def test_enumeration_question_takes_whole_source_not_top_k():
    """「列出所有项目」必须拿到该来源的全部卡片，top_k=3 不可能列全。"""
    pipeline = build_big(top_k=3)
    hits = pipeline.retrieve("把简历里所有项目列一个清单出来")

    assert len(hits) == len(RESUME), f"只拿到 {len(hits)} 张，应为该来源全部 {len(RESUME)} 张"
    assert {h.chunk.source for h in hits} == {"简历"}


def test_normal_question_still_uses_top_k():
    pipeline = build_big(top_k=3)
    hits = pipeline.retrieve("牛皮怎么处理")
    assert len(hits) == 3


def test_enumeration_plan_widens_context_and_output_budget():
    pipeline = build_big()
    plan = pipeline.plan("把简历里所有项目列一个清单出来")

    assert plan.mode == "enumerate"
    assert plan.max_context_chars >= 8000, "清单模式上下文预算必须放开，否则 42 张卡装不下"
    assert plan.max_tokens >= 3000, "输出上限 800 只能写出 4 个项目"


def test_normal_plan_keeps_default_budgets():
    pipeline = build_big()
    plan = pipeline.plan("皮影戏是怎么做的")
    assert plan.mode == "qa"
    assert plan.max_tokens <= 1500


def test_enumeration_prompt_forbids_omission():
    pipeline = build_big()
    plan = pipeline.plan("把简历里所有项目列一个清单出来")
    prompt = plan.system_prompt
    assert "逐条" in prompt or "全部" in prompt
    assert "省略" in prompt or "合并" in prompt
    assert "不要编造" in prompt or "不要臆造" in prompt


def test_enumeration_falls_back_when_source_is_too_large():
    """超出上限的长文档不能整篇塞进去，必须退回大 top_k，避免上下文爆掉。"""
    many = [Chunk(id=f"m{i}", text=f"长文档第{i}段，讲的是同一个主题。", source="长文档") for i in range(60)]
    retriever = HybridRetriever(embedder=HashingEmbedder(), keyword_weight=0.6)
    retriever.index(many)
    pipeline = RAGPipeline(retriever=retriever, llm=RecordingLLM(), top_k=3)

    hits = pipeline.retrieve("把长文档里所有内容列一个清单出来")
    assert len(hits) <= pipeline.enum_max_source_chunks, "超长来源应退化为有限条数"
    assert len(hits) >= 10, "退化后也要比普通 top_k 拿得多"


def test_enumeration_passes_wider_max_tokens_to_llm():
    llm = RecordingLLM()
    pipeline = build_big(llm=llm)
    asyncio.run(pipeline.answer("把简历里所有项目列一个清单出来"))
    assert llm.max_tokens_seen[0] and llm.max_tokens_seen[0] >= 3000


def test_enumeration_fallback_keeps_document_order():
    """退化（来源过长）时命中的卡片也要按原文顺序返回。

    清单型问题常要求「按时间从近到远」，相似度顺序和原文时间线没关系，
    照抄就会把 2015 年的项目排到 2021 年前面。
    """
    many = [
        Chunk(id=f"m{i}", text=f"第{i}段：这是长文档里的第 {i} 段内容，主题相同。", source="长文档")
        for i in range(80)
    ]
    retriever = HybridRetriever(embedder=HashingEmbedder(), keyword_weight=0.6)
    retriever.index(many)
    pipeline = RAGPipeline(retriever=retriever, llm=RecordingLLM(), top_k=3)

    hits = pipeline.retrieve("把长文档里所有内容列一个清单出来")
    positions = [int(h.chunk.id[1:]) for h in hits]

    assert positions == sorted(positions), f"退化后必须按原文顺序：{positions}"


def test_enumeration_prompt_limits_each_entry():
    """每条不设上限时，模型会把 4 个项目写成 6600 字，后面的项目直接被输出上限截掉。"""
    pipeline = build_big()
    plan = pipeline.plan("把简历里所有项目列一个清单出来")
    assert "字以内" in plan.system_prompt, "必须给单条长度设上限，否则长清单会被截断"


def test_list_style_question_on_single_card_source_stays_qa():
    """列举句式但来源只有一张卡时，不能按清单模式处理。

    反例：「秦腔有哪些角色行当」含「哪些」，但库里就一张卡。
    走清单模式只会让回答变成长清单，还白白丢掉语音播报。
    """
    retriever = HybridRetriever(embedder=HashingEmbedder(), keyword_weight=0.6)
    retriever.index([Chunk(id="solo", text="秦腔的角色行当分四生六旦二净一丑。", source="秦腔")])
    pipeline = RAGPipeline(retriever=retriever, llm=RecordingLLM(), top_k=5)

    plan = pipeline.plan("秦腔有哪些角色行当？")
    assert plan.mode == "qa", "单卡来源不该走清单模式"
    assert len(plan.hits) == 1, "那一张卡还是必须被检索到"
    assert not pipeline.wants_enumeration("秦腔有哪些角色行当？")


def test_wants_enumeration_tracks_plan_decision():
    """Session 靠这个判断要不要合成语音，必须和 plan() 完全一致。"""
    pipeline = build_big(top_k=3)
    assert pipeline.wants_enumeration("把简历里所有项目列一个清单出来")
    assert not pipeline.wants_enumeration("皮影戏是怎么做的")

