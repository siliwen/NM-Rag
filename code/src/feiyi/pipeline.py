"""RAG 编排：问题 → 检索 → 拼 prompt → 大模型流式生成。

设计要点：
- 只用检索到的资料回答，找不到就明确说不知道，不编造
- prompt 里强制标注资料来源，便于人工核查
- **清单型问题**（「把所有项目列出来」这类）走全量取卡 + 放宽输出上限：
  top_k 只取几段，对「要全」的问题天然会漏；而且输出长度不够时，
  模型会把 4 个项目写完就停，看起来像「只找到 4 个」
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Sequence

from .clients.llm import LLMClient
from .rag.retriever import HybridRetriever
from .rag.store import ScoredChunk

SYSTEM_PROMPT = """你是知识库问答助手，依据【参考资料】回答用户的问题。

回答要求：
1. 只能依据【参考资料】作答；资料里没有的内容，直接说明"资料里没有记载"，不要编造。
2. 不要提及"资料""检索""文档"这类系统词汇，直接讲内容。
3. 涉及工序、年代、地名、数量等具体信息时，必须与资料一致。
4. 用清楚的中文回答，先把问题讲明白，再补必要的细节。"""

# 清单型问题专用：这类问题要的是「全」，不能按讲解口吻压缩
ENUMERATION_PROMPT = """你是信息整理助手。请依据【参考资料】把用户要的内容完整列出。

要求：
1. 逐条列出，不要省略、不要合并、不要只给代表性例子。资料里有几条就列几条。
2. 每条保留资料里的原始名称、时间、数字，不要改写名称，也不要编造资料里没有的名称或数据。
3. 某项信息资料里没有，就写"资料未提及"，不要推测补全。
4. 用户要求排序时（例如按时间从近到远），严格按资料里的时间排。
5. **每条控制在 150 字以内**：简介一句话说清是什么，成果一句话说清交付了什么。
   清单的价值是能一眼扫完，不要展开成长篇叙述。
6. 不要在开头结尾加寒暄，也不要写"以上是全部"这类总结。"""

# 清单型问题的信号词：命中这些词说明用户要「全量」，按 top_k 取几段必然漏
ENUM_SIGNALS = (
    "所有",
    "全部",
    "清单",
    "列表",
    "罗列",
    "列出",
    "列举",
    "有哪些",
    "哪些",
    "哪几个",
    "汇总",
    "一共",
    "总共",
    "逐个",
    "一一",
)

# 清单模式的预算：实测 46 张卡 / 8400 字需要 16000 字上下文；
# 输出给 6000 是保险值——13 个项目每条 150 字约 2000 字，模型不听话写长了也不会被截断
ENUM_CONTEXT_CHARS = 16000
ENUM_MAX_TOKENS = 6000
# 单个来源超过这个卡片数就不整篇塞了（上下文会爆），退回大 top_k。
# 简历类文档切出来 40~60 张很常见，别设太小，否则又会退化成"只看一部分"
ENUM_MAX_SOURCE_CHUNKS = 60
ENUM_FALLBACK_TOP_K = 20


def is_enumeration(question: str) -> bool:
    """是不是「要全量」的问题（列出/全部/清单/汇总…）。"""
    q = question or ""
    return any(sig in q for sig in ENUM_SIGNALS)


@dataclass
class RetrievalPlan:
    """一次问答的检索与生成计划：清单型和普通问答走不同预算。"""

    mode: str  # "qa" | "enumerate"
    hits: list[ScoredChunk] = field(default_factory=list)
    system_prompt: str = SYSTEM_PROMPT
    max_context_chars: int = 1500
    max_tokens: int = 800


@dataclass
class Answer:
    text: str
    sources: list[ScoredChunk] = field(default_factory=list)


class RAGPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        llm: LLMClient,
        top_k: int = 3,
        max_context_chars: int = 1500,
        max_tokens: int = 800,
        enum_context_chars: int = ENUM_CONTEXT_CHARS,
        enum_max_tokens: int = ENUM_MAX_TOKENS,
        enum_max_source_chunks: int = ENUM_MAX_SOURCE_CHUNKS,
        enum_fallback_top_k: int = ENUM_FALLBACK_TOP_K,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.max_context_chars = max_context_chars
        self.max_tokens = max_tokens
        self.enum_context_chars = enum_context_chars
        self.enum_max_tokens = enum_max_tokens
        self.enum_max_source_chunks = enum_max_source_chunks
        self.enum_fallback_top_k = enum_fallback_top_k

    # ---------- 检索 ----------
    def plan(self, question: str) -> RetrievalPlan:
        if is_enumeration(question):
            whole = self._retrieve_whole_source(question)
            # 只有「该来源的卡片比普通 top_k 还多」时才值得走清单模式。
            # 反例：「秦腔有哪些角色行当」也是列举，但库里就 1 张卡——
            # 这时按清单模式处理只会让回答变成长清单，还白白丢掉语音播报。
            if len(whole) > self.top_k:
                return RetrievalPlan(
                    mode="enumerate",
                    hits=whole,
                    system_prompt=ENUMERATION_PROMPT,
                    max_context_chars=self.enum_context_chars,
                    max_tokens=self.enum_max_tokens,
                )
        return RetrievalPlan(
            mode="qa",
            hits=self.retriever.search(question, top_k=self.top_k),
            system_prompt=SYSTEM_PROMPT,
            max_context_chars=self.max_context_chars,
            max_tokens=self.max_tokens,
        )

    def wants_enumeration(self, question: str) -> bool:
        """这次问答会不会走清单模式。

        调用方（Session）需要据此决定要不要合成语音：清单型问题动辄几十条、
        逐句合成几分钟，用听的方式列清单也没意义。判断必须复用 plan()，
        不能再按问题文本各判一次——「秦腔有哪些角色行当」也是列举句式，
        但它只命中一张卡，按普通问答走，语音不能丢。
        """
        return self.plan(question).mode == "enumerate"

    def retrieve(self, question: str) -> list[ScoredChunk]:
        return self.plan(question).hits

    def _retrieve_whole_source(self, question: str) -> list[ScoredChunk]:
        """清单型问题：先定位最相关的来源，再把这个来源的卡片按原顺序全取出来。

        为什么不能只调大 top_k：top_k=20 时命中的 20 张可能全挤在同一段，
        另一些项目一张都没有；而「按来源全量」能保证同一份文档不残缺。
        """
        seed = self.retriever.search(question, top_k=1)
        if not seed:
            return []
        source = seed[0].chunk.source
        chunks = getattr(self.retriever.store, "chunks", lambda: [])()
        if not source:
            return self._ordered_search(question, chunks)

        pool = [c for c in chunks if c.source == source]
        if len(pool) <= 1:
            # 该来源就一张卡：普通检索已经能拿全，不必走清单模式（也就不必丢语音）
            return [ScoredChunk(chunk=c, score=seed[0].score) for c in pool]
        if len(pool) > self.enum_max_source_chunks:
            # 超长文档整篇塞进去会爆上下文，退回有限条数（仍然比普通 top_k 多）
            return self._ordered_search(question, chunks)
        # 保持来源内的原始顺序（PDF 逐页入库，顺序等于原文顺序）
        return [ScoredChunk(chunk=c, score=seed[0].score) for c in pool]

    def _ordered_search(self, question: str, chunks: Sequence) -> list[ScoredChunk]:
        """退化检索：取相似度 top-N，但**按原文顺序**返回。

        为什么必须重排：清单型问题常要求「按时间从近到远」，而相似度排出来的
        顺序和原文时间线毫无关系，模型照抄就会把 2015 年的项目排到 2021 年前面。
        """
        hits = self.retriever.search(question, top_k=self.enum_fallback_top_k)
        order = {c.id: index for index, c in enumerate(chunks)}
        return sorted(hits, key=lambda hit: order.get(hit.chunk.id, 10**9))

    # ---------- 拼 prompt ----------
    def build_messages(
        self,
        question: str,
        hits: Sequence[ScoredChunk],
        system_prompt: str | None = None,
        max_context_chars: int | None = None,
    ) -> list[dict]:
        budget = self.max_context_chars if max_context_chars is None else max_context_chars
        blocks: list[str] = []
        used = 0
        for index, hit in enumerate(hits, start=1):
            text = hit.chunk.text.strip()
            if used + len(text) > budget:
                text = text[: max(0, budget - used)]
            if not text:
                break
            used += len(text)
            label = hit.chunk.source or f"资料{index}"
            # PDF 卡片带页码：让大模型知道内容出自哪一页，溯源时口径一致
            page = hit.chunk.meta.get("page")
            if page:
                label = f"{label} 第{page}页"
            blocks.append(f"（{index}｜来源：{label}）{text}")
            if used >= budget:
                break

        context = "\n".join(blocks) if blocks else "（本次没有检索到相关资料）"
        user = f"【参考资料】\n{context}\n\n【用户问题】{question}"
        return [
            {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]

    # ---------- 生成 ----------
    async def stream(
        self, question: str, plan: RetrievalPlan | None = None
    ) -> AsyncIterator[tuple[str, Any]]:
        """产出 ("sources", list[ScoredChunk]) 与 ("token", str) 两类事件。

        plan 可由调用方预先算好传进来：Session 需要先知道是不是清单模式才能
        决定要不要语音，若这里再算一次等于白检索两遍。
        """
        plan = plan or self.plan(question)
        yield ("sources", plan.hits)
        messages = self.build_messages(question, plan.hits, plan.system_prompt, plan.max_context_chars)
        async for token in self.llm.stream(messages, max_tokens=plan.max_tokens):
            yield ("token", token)

    async def answer(self, question: str) -> Answer:
        plan = self.plan(question)
        messages = self.build_messages(question, plan.hits, plan.system_prompt, plan.max_context_chars)
        text = await self.llm.complete(messages, max_tokens=plan.max_tokens)
        return Answer(text=text, sources=plan.hits)
