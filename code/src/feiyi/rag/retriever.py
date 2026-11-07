"""混合检索：向量召回 + 关键词（BM25）打分，可选重排。

为什么必须混合：非遗场景里大量专名（皮影、秦腔、蓝印花布），
纯向量容易把「读音相近但不同」的条目召回，BM25 能兜住字面精确匹配。
"""
from __future__ import annotations

import math
import re
from typing import Callable, Sequence

from .chunking import Chunk
from .embedder import Embedder, _features
from .store import InMemoryVectorStore, ScoredChunk, VectorStore

_LATIN = re.compile(r"[a-zA-Z0-9]+")
_CJK = re.compile(r"[\u4e00-\u9fff]")

Reranker = Callable[[str, list[ScoredChunk]], list[ScoredChunk]]


def tokenize(text: str) -> list[str]:
    """中文二元组 + 英文/数字整词。"""
    tokens = [w.lower() for w in _LATIN.findall(text or "")]
    zh = "".join(_CJK.findall(text or ""))
    tokens.extend(zh[i : i + 2] for i in range(len(zh) - 1))
    return tokens


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class HybridRetriever:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore | None = None,
        keyword_weight: float = 0.6,
        reranker: Reranker | None = None,
        k1: float = 1.5,
        b: float = 0.75,
        keyword_saturation: float = 3.0,
    ) -> None:
        self.embedder = embedder
        # 注意：不能用 `store or InMemoryVectorStore()`，空 store 的 __len__ 为 0 会被判定为假
        self.store: VectorStore = store if store is not None else InMemoryVectorStore()
        self.keyword_weight = _clamp(keyword_weight, 0.0, 1.0)
        self.keyword_saturation = max(0.1, float(keyword_saturation))
        self.reranker = reranker
        self.k1 = k1
        self.b = b
        self._doc_tokens: list[list[str]] = []
        self._df: dict[str, int] = {}
        self._indexed: set[str] = set()
        self._avgdl: float = 0.0
        # 持久化库重启后自带卡片，但 BM25 索引必须靠这里重建，
        # 否则向量分还在、关键词分全为 0，检索质量会无声退化。
        self.rebuild()

    # ---------- 索引 ----------
    def _index_tokens(self, chunk: Chunk) -> None:
        # 来源标题同样入索引：非遗资料里「皮影制皮」这类小标题往往就是条目名，
        # 让标题参与关键词打分能显著提升专名召回率。
        tokens = tokenize(chunk.text) + tokenize(chunk.source)
        self._doc_tokens.append(tokens)
        self._indexed.add(chunk.id)
        for token in set(tokens):
            self._df[token] = self._df.get(token, 0) + 1

    def _refresh_avgdl(self) -> None:
        total = sum(len(t) for t in self._doc_tokens)
        self._avgdl = total / len(self._doc_tokens) if self._doc_tokens else 0.0

    def rebuild(self) -> int:
        """从 store 重建关键词索引，返回文档数。重启恢复、换库后调用。"""
        self._doc_tokens.clear()
        self._df.clear()
        self._indexed.clear()
        for chunk in self.store.chunks():
            self._index_tokens(chunk)
        self._refresh_avgdl()
        return len(self._doc_tokens)

    def index(self, chunks: Sequence[Chunk]) -> int:
        fresh = [c for c in chunks if c.id not in self._indexed]
        if not fresh:
            return 0
        vectors = self.embedder.embed([c.text for c in fresh])
        self.store.add(fresh, vectors)
        for chunk in fresh:
            self._index_tokens(chunk)
        self._refresh_avgdl()
        return len(fresh)

    # ---------- 检索 ----------
    def _bm25(self, query: str) -> dict[str, float]:
        n_docs = len(self._doc_tokens)
        if not n_docs:
            return {}
        q_tokens = set(tokenize(query))
        if not q_tokens:
            return {}
        scores: dict[str, float] = {}
        for chunk, tokens in zip(self.store.chunks(), self._doc_tokens):
            length = len(tokens) or 1
            freq: dict[str, int] = {}
            for token in tokens:
                freq[token] = freq.get(token, 0) + 1
            score = 0.0
            for token in q_tokens:
                df = self._df.get(token, 0)
                if not df:
                    continue
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                tf = freq.get(token, 0)
                if not tf:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * length / (self._avgdl or 1))
                score += idf * (tf * (self.k1 + 1)) / denom
            if score > 0:
                scores[chunk.id] = score
        return scores

    def search(self, query: str, top_k: int = 3) -> list[ScoredChunk]:
        """返回 top_k 张卡片。数据量在万级以内时用全量打分，简单且可控。"""
        if not len(self.store):
            return []

        query = (query or "").strip()
        if not query:
            return []

        # BGE 中文系列是「非对称检索」：查询侧加指令、文档侧不加。
        # 未实现 embed_query 的向量器（如 hash 桩）退回普通 embed。
        embed_query = getattr(self.embedder, "embed_query", None)
        if callable(embed_query):
            query_vec = embed_query(query)
        else:
            query_vec = self.embedder.embed([query])[0]
        vector_hits = self.store.search(query_vec, top_k=len(self.store))
        vec_scores = {hit.chunk.id: hit.score for hit in vector_hits}
        keyword_scores = self._bm25(query)

        weight = self.keyword_weight
        merged: list[ScoredChunk] = []
        for chunk in self.store.chunks():
            # 向量用原始余弦（本身已在 [-1,1]，直接截断到 [0,1]）；
            # 不能用「除以最大值」，否则最差的一批也会被抬到 1.0，把关键词信号压掉。
            vec_norm = max(0.0, vec_scores.get(chunk.id, 0.0))
            # 关键词分不能按最大值归一化：那会让「任何有字样重叠的查询」的最优结果
            # 都固定拿到 1.0，越界问题（食堂几点开门）也会得 0.6 以上的高分，
            # 无法靠分数阈值判断该不该拒答。改用饱和函数保留绝对量级。
            kw_raw = keyword_scores.get(chunk.id, 0.0)
            kw_norm = kw_raw / (kw_raw + self.keyword_saturation) if kw_raw > 0 else 0.0
            final = (1 - weight) * vec_norm + weight * kw_norm
            merged.append(
                ScoredChunk(
                    chunk=chunk,
                    score=final,
                    detail={"vector": round(vec_norm, 4), "keyword": round(kw_norm, 4)},
                )
            )

        merged.sort(key=lambda s: s.score, reverse=True)
        if self.reranker is not None:
            merged = self.reranker(query, merged)
        return merged[: max(1, top_k)]

    def __len__(self) -> int:
        return len(self.store)
