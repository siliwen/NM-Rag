"""LLM 接入自检：验证 .env 里的密钥能不能真正调通。

用法：
    .venv\\Scripts\\python.exe scripts\\check_llm.py
    .venv\\Scripts\\python.exe scripts\\check_llm.py --model qwen-max

做三件事：
    1. 打印配置（密钥打码，不泄露）
    2. 非流式请求一次，测端到端延迟
    3. 流式请求一次，测首字延迟（数字人场景的关键指标）

任何失败都显式抛出并以非 0 退出（NO-006：不静默跳过）。
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from feiyi.clients.llm import OpenAICompatLLM  # noqa: E402
from feiyi.config import load_dotenv  # noqa: E402

PROMPT = "用一句话介绍皮影戏，不超过30字。"


def mask(key: str) -> str:
    if not key:
        return "(未配置)"
    if len(key) <= 10:
        return key[:2] + "***"
    return f"{key[:6]}…{key[-4:]}（共 {len(key)} 位）"


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="", help="覆盖 .env 里的 LLM_MODEL")
    parser.add_argument("--base-url", default="", help="覆盖 .env 里的 LLM_BASE_URL")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")

    import os

    base_url = args.base_url or os.environ.get("LLM_BASE_URL", "")
    api_key = os.environ.get("LLM_API_KEY", "")
    model = args.model or os.environ.get("LLM_MODEL", "")

    print("=" * 56)
    print("LLM 接入自检")
    print("=" * 56)
    print(f"  base_url : {base_url or '(空)'}")
    print(f"  api_key  : {mask(api_key)}")
    print(f"  model    : {model or '(空)'}")

    if not api_key:
        print("\n[失败] .env 里 LLM_API_KEY 为空。")
        print("       请把百炼密钥填进 code\\.env 的 LLM_API_KEY= 后面，再重跑本脚本。")
        return 1
    if not base_url:
        print("\n[失败] LLM_BASE_URL 为空，无法请求。")
        return 1

    client = OpenAICompatLLM(base_url=base_url, api_key=api_key, model=model)
    messages = [
        {"role": "system", "content": "你是非遗讲解员，回答简洁准确。"},
        {"role": "user", "content": PROMPT},
    ]

    # 1) 非流式
    print("\n[1/2] 非流式请求…")
    t0 = time.perf_counter()
    try:
        text = await client.complete(messages)
    except Exception as exc:  # noqa: BLE001 - 显式上报任何失败
        print(f"[失败] {type(exc).__name__}: {exc}")
        return 1
    cost = (time.perf_counter() - t0) * 1000
    print(f"  OK  {cost:.0f} ms")
    print(f"  回答：{text.strip()[:200]}")

    # 2) 流式 + 首字延迟
    print("\n[2/2] 流式请求（测首字延迟）…")
    t0 = time.perf_counter()
    first = None
    pieces: list[str] = []
    try:
        async for piece in client.stream(messages):
            if first is None:
                first = (time.perf_counter() - t0) * 1000
            pieces.append(piece)
    except Exception as exc:  # noqa: BLE001
        print(f"[失败] {type(exc).__name__}: {exc}")
        return 1
    total = (time.perf_counter() - t0) * 1000
    print(f"  首字 {first:.0f} ms / 整轮 {total:.0f} ms / 分片 {len(pieces)} 个")
    print(f"  回答：{''.join(pieces).strip()[:200]}")

    print("\n" + "=" * 56)
    print("结论：密钥可用，编排层可直接接入。")
    print(f"     首字延迟 {first:.0f} ms（数字人场景建议 < 800 ms）")
    print("=" * 56)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
