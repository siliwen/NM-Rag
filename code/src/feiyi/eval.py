"""检索质量评测核心：脚本与 HTTP 接口共用，避免两处实现各写一套。

评测集约定（jsonl，每行一条）：
    {"q": "问题", "expect": "期望命中的 source", "level": "L1|L2|L3|L4", "note": "..."}

级别含义：
    L1 字面匹配 —— 问题里含专名，考的是关键词召回
    L2 同义改写 —— 观众口语提问，考的是语义向量
    L3 跨段推理 —— 需要关联多个概念
    L4 越界问题 —— 完全无关，expect 为 null
    L5 相关无事实 —— 话题相关但库里没有这个事实（如票价），expect 为 null

L4 与 L5 都不计入命中率分母。两者要分开看：L4 是检索该「不召回」，
L5 是检索应该召回（主题相关）、由大模型去说「资料里没有记载」。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .rag.chunking import Chunk
from .rag.retriever import HybridRetriever

LEVEL_TITLES = {
    "L1": "字面匹配",
    "L2": "同义改写",
    "L3": "跨段推理",
    "L4": "完全越界",
    "L5": "相关无事实",
}
GRADED_LEVELS = ("L1", "L2", "L3")
OOS_LEVELS = ("L4", "L5")


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append(json.loads(line))
    return rows


def build_corpus(path: Path) -> list[Chunk]:
    return [
        Chunk(id=f"c{i:03d}", text=row["text"], source=row.get("source", ""))
        for i, row in enumerate(read_jsonl(path))
    ]


def evaluate(
    embedder: Any,
    chunks: list[Chunk],
    qa: list[dict],
    keyword_weight: float | None = None,
    top_k: int = 3,
) -> dict[str, Any]:
    semantic = hasattr(embedder, "model_id")
    weight = (0.2 if semantic else 0.6) if keyword_weight is None else keyword_weight
    started = time.perf_counter()
    retriever = HybridRetriever(embedder=embedder, keyword_weight=weight)
    retriever.index(chunks)
    index_ms = (time.perf_counter() - started) * 1000

    stats: dict[str, list[int]] = {k: [0, 0] for k in LEVEL_TITLES}
    mrr_sum = 0.0
    mrr_n = 0
    hit3 = 0
    oos_scores: dict[str, list[float]] = {k: [] for k in OOS_LEVELS}
    failures: list[dict[str, Any]] = []
    latencies: list[float] = []

    for item in qa:
        level = item["level"]
        expect = item.get("expect")
        t0 = time.perf_counter()
        hits = retriever.search(item["q"], top_k=top_k)
        latencies.append((time.perf_counter() - t0) * 1000)
        names = [h.chunk.source for h in hits]
        top_score = hits[0].score if hits else 0.0

        if level in OOS_LEVELS:
            stats[level][1] += 1
            oos_scores[level].append(top_score)
            continue

        stats[level][1] += 1
        if expect in names:
            hit3 += 1
        if names and names[0] == expect:
            stats[level][0] += 1
            mrr_sum += 1.0
        elif expect in names:
            rank = names.index(expect) + 1
            mrr_sum += 1.0 / rank
            failures.append(
                {"q": item["q"], "level": level, "expect": expect,
                 "got": names, "rank": rank, "note": item.get("note", "")}
            )
        else:
            failures.append(
                {"q": item["q"], "level": level, "expect": expect,
                 "got": names, "rank": 0, "note": item.get("note", "")}
            )
        mrr_n += 1

    total_hit = sum(stats[k][0] for k in GRADED_LEVELS)
    total_n = sum(stats[k][1] for k in GRADED_LEVELS)
    return {
        "embedder": getattr(embedder, "model_id", type(embedder).__name__),
        "dim": getattr(embedder, "dim", 0),
        "semantic": semantic,
        "keyword_weight": weight,
        "cards": len(chunks),
        "questions": len(qa),
        "hit1": round(total_hit / total_n, 4) if total_n else 0.0,
        "hit3": round(hit3 / total_n, 4) if total_n else 0.0,
        "mrr": round(mrr_sum / mrr_n, 4) if mrr_n else 0.0,
        "by_level": {
            k: {
                "title": LEVEL_TITLES[k],
                "hit": v[0],
                "total": v[1],
                "rate": round(v[0] / v[1], 4) if v[1] else 0.0,
            }
            for k, v in stats.items()
        },
        "oos": {
            k: {
                "peak": round(max(v), 4) if v else 0.0,
                "avg": round(sum(v) / len(v), 4) if v else 0.0,
                "n": len(v),
            }
            for k, v in oos_scores.items()
        },
        "index_ms": round(index_ms, 1),
        "query_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        "failures": failures,
    }
