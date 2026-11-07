"""RAG 编排：问题 → 检索 → 拼 prompt → 大模型流式生成。

设计要点：
- 只用检索到的资料回答，找不到就明确说不知道，不编造（非遗内容必须准确）
- prompt 里强制标注资料来源，便于人工核查
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Sequence

from .clients.llm import LLMClient
from .rag.retriever import HybridRetriever
from .rag.store import ScoredChunk

SYSTEM_PROMPT = """你是非物质文化遗产项目的数字讲解员，用传承人的身份向观众介绍这项非遗。

回答要求：
1. 只能依据【参考资料】作答；资料里没有的内容，直接说明"这部分资料里还没有记载"，不要编造。
2. 用口语化的短句讲解，像面对面聊天，避免书面语和长句。
3. 单次回答控制在 120 字以内，先回答问题核心，再补充一到两个细节。
4. 不要提及"资料""检索""文档"这类系统词汇，直接讲内容。
5. 涉及工序、年代、地名等具体信息时，必须与资料一致。"""


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
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.max_context_chars = max_context_chars

    def retrieve(self, question: str) -> list[ScoredChunk]:
        return self.retriever.search(question, top_k=self.top_k)

    def build_messages(self, question: str, hits: Sequence[ScoredChunk]) -> list[dict]:
        blocks: list[str] = []
        used = 0
        for index, hit in enumerate(hits, start=1):
            text = hit.chunk.text.strip()
            if used + len(text) > self.max_context_chars:
                text = text[: max(0, self.max_context_chars - used)]
            if not text:
                break
            used += len(text)
            label = hit.chunk.source or f"资料{index}"
            # PDF 卡片带页码：让大模型知道内容出自哪一页，溯源时口径一致
            page = hit.chunk.meta.get("page")
            if page:
                label = f"{label} 第{page}页"
            blocks.append(f"（{index}｜来源：{label}）{text}")
            if used >= self.max_context_chars:
                break

        context = "\n".join(blocks) if blocks else "（本次没有检索到相关资料）"
        user = f"【参考资料】\n{context}\n\n【观众问题】{question}"
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]

    async def stream(self, question: str) -> AsyncIterator[tuple[str, Any]]:
        """产出 ("sources", list[ScoredChunk]) 与 ("token", str) 两类事件。"""
        hits = self.retrieve(question)
        yield ("sources", hits)
        messages = self.build_messages(question, hits)
        async for token in self.llm.stream(messages):
            yield ("token", token)

    async def answer(self, question: str) -> Answer:
        hits = self.retrieve(question)
        messages = self.build_messages(question, hits)
        text = await self.llm.complete(messages)
        return Answer(text=text, sources=hits)
