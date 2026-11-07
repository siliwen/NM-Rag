"""模型延迟对比：同一问题跑多个 qwen 模型，比较首字延迟与总延迟。

数字人场景的核心指标是**首字延迟**（观众问完到数字人张嘴的时间），
而不是总耗时。本脚本每个模型多跑几次取中位数，避免单次抖动误导。

用法：
    .venv\\Scripts\\python.exe scripts\\bench_llm.py
    .venv\\Scripts\\python.exe scripts\\bench_llm.py --models qwen-plus,qwen-turbo --rounds 3
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from feiyi.clients.llm import OpenAICompatLLM  # noqa: E402
from feiyi.config import load_dotenv  # noqa: E402

PROMPT = "用口语讲一下皮影戏的牛皮要怎么处理，三句话以内。"


async def bench(model: str, base_url: str, api_key: str, rounds: int) -> dict:
    client = OpenAICompatLLM(base_url=base_url, api_key=api_key, model=model, timeout=60)
    messages = [
        {"role": "system", "content": "你是非遗讲解员，用口语化的短句回答，不超过三句。"},
        {"role": "user", "content": PROMPT},
    ]
    firsts: list[float] = []
    totals: list[float] = []
    text = ""
    for _ in range(rounds):
        t0 = time.perf_counter()
        first = None
        pieces: list[str] = []
        try:
            async for piece in client.stream(messages):
                if first is None:
                    first = (time.perf_counter() - t0) * 1000
                pieces.append(piece)
        except Exception as exc:  # noqa: BLE001 - 显式上报
            return {"model": model, "error": f"{type(exc).__name__}: {exc}"}
        totals.append((time.perf_counter() - t0) * 1000)
        firsts.append(first or 0.0)
        text = "".join(pieces)
    return {
        "model": model,
        "first_med": statistics.median(firsts),
        "first_min": min(firsts),
        "total_med": statistics.median(totals),
        "chars": len(text),
        "sample": text.strip().replace("\n", " ")[:60],
    }


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="qwen-turbo,qwen-plus,qwen-max")
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    import os

    base_url = os.environ.get("LLM_BASE_URL", "")
    api_key = os.environ.get("LLM_API_KEY", "")
    if not api_key:
        print("[失败] LLM_API_KEY 为空，先填 code\\.env")
        return 1

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    print(f"基准问题：{PROMPT}")
    print(f"每个模型跑 {args.rounds} 次取中位数\n")

    rows = []
    for model in models:
        print(f"  测试 {model} …", flush=True)
        rows.append(await bench(model, base_url, api_key, args.rounds))

    print("\n" + "=" * 74)
    print(f"{'模型':<18}{'首字中位':>10}{'首字最快':>10}{'总耗时中位':>12}{'字数':>7}")
    print("-" * 74)
    ok = [r for r in rows if "error" not in r]
    for r in rows:
        if "error" in r:
            print(f"{r['model']:<18}{'失败':>10}  {r['error'][:40]}")
        else:
            print(
                f"{r['model']:<18}{r['first_med']:>8.0f}ms{r['first_min']:>8.0f}ms"
                f"{r['total_med']:>10.0f}ms{r['chars']:>7}"
            )
    print("=" * 74)
    if ok:
        best = min(ok, key=lambda r: r["first_med"])
        print(f"首字最快：{best['model']}（{best['first_med']:.0f} ms）")
        print(f"数字人建议：首字 < 800 ms 才算自然，> 1500 ms 观众会觉得卡。")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
