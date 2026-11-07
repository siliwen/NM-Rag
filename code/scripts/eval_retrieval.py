"""检索质量评测：把「我觉得答得还行」换成可比的数字。

用法：
    python scripts/eval_retrieval.py                      # 用 .env 里的 EMBEDDER
    python scripts/eval_retrieval.py --embedder hash      # 只看测试桩基线
    python scripts/eval_retrieval.py --embedder bge-small-zh
    python scripts/eval_retrieval.py --compare            # 测试桩 vs 语义模型并排对比

指标说明：
    hit@1 / hit@3  正确答案是否落在第 1 / 前 3 位
    MRR            正确卡片名次的倒数均值，越接近 1 越好
    L4 最高分      越界问题拿到的最高分；越低越不容易让模型硬答
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from feiyi.config import load_dotenv  # noqa: E402
from feiyi.eval import build_corpus, evaluate, read_jsonl  # noqa: E402
from feiyi.rag.embedder import build_embedder  # noqa: E402

CORPUS = ROOT / "data" / "corpus" / "feiyi_demo.jsonl"
EVAL = ROOT / "data" / "eval" / "qa_set.jsonl"


def print_report(result: dict) -> None:
    kind = "语义向量" if result["semantic"] else "字符哈希桩"
    print(f"\n=== {result['embedder']}（{result['dim']} 维 · {kind} · 关键词权重 {result['keyword_weight']}） ===")
    print(f"  入库 {result['cards']} 张 {result['index_ms']:.0f} ms | 平均检索 {result['query_ms']:.0f} ms")
    for level in ("L1", "L2", "L3"):
        item = result["by_level"][level]
        rate = item["rate"] * 100
        bar = "█" * round(rate / 5) if item["total"] else ""
        print(f"  {level} {item['title']}  {item['hit']}/{item['total']}  {rate:5.1f}%  {bar}")
    print(f"  总 hit@1 {result['hit1']*100:.1f}%  hit@3 {result['hit3']*100:.1f}%  MRR {result['mrr']:.3f}")
    for level in ("L4", "L5"):
        item = result["by_level"][level]
        oos = result["oos"][level]
        print(f"  {level} {item['title']}（{item['total']} 题）最高分 峰值 {oos['peak']:.3f} / 平均 {oos['avg']:.3f}")
    if result["failures"]:
        print(f"  未命中 {len(result['failures'])} 题：")
        for f in result["failures"]:
            if f["rank"] == 1:
                continue
            where = f"第 {f['rank']} 位" if f["rank"] else "完全未命中"
            print(f"  [{f['level']}] {f['q']} → 应为「{f['expect']}」，{where}，实得 {f['got']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedder", default="", help="hash / bge-small-zh / bge-m3 / HF 模型 ID")
    parser.add_argument("--compare", action="store_true", help="测试桩与语义模型并排对比")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--keyword-weight", type=float, default=None,
        help="关键词权重；不填则按向量能力自动取值（语义 0.2 / hash 0.6）",
    )
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    chunks = build_corpus(CORPUS)
    qa = read_jsonl(EVAL)
    counts = {lvl: sum(1 for x in qa if x["level"] == lvl) for lvl in ("L1", "L2", "L3", "L4", "L5")}
    print(f"评测集：{len(chunks)} 张卡片 / {len(qa)} 道题 "
          f"(L1 {counts['L1']} | L2 {counts['L2']} | L3 {counts['L3']} | L4 {counts['L4']} | L5 {counts['L5']})")

    if args.compare:
        names = ["hash", args.embedder or "bge-small-zh"]
    else:
        names = [args.embedder or os.environ.get("EMBEDDER", "hash")]

    for name in names:
        try:
            emb = build_embedder(name)
        except Exception as exc:
            print(f"\n[跳过 {name}] {exc}")
            continue
        print_report(evaluate(emb, chunks, qa, keyword_weight=args.keyword_weight, top_k=args.top_k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
