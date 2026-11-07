"""端到端冒烟：入库 → 检索 → 问答流。服务需先启动。

用法：
    python scripts/smoke.py
    python scripts/smoke.py --base-url http://127.0.0.1:8020

判定标准（与 AGENT.md 的指标一致）：
    - 检索首位命中期望内容
    - 问答事件流出现 sources / token / done
    - 记录首字耗时，供后续对比
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import time

import httpx

SAMPLE = (
    "皮影戏又称灯影戏，起源于西汉，盛行于唐宋。"
    "选皮多用牛皮或驴皮，制皮时要把皮子浸泡刮薄晾干，使皮面平整透光。"
    "镂刻是最关键的一步，艺人用数十把刻刀在皮面上刻出人物轮廓与纹样。"
    "秦腔是陕西地方戏曲，唱腔高亢激越。"
)
QUESTION = "皮影的牛皮是怎么处理的"
EXPECT_KEYWORD = "牛皮"

# 冒烟数据用带时间戳的来源，跑完自动删除，避免污染持久化知识库
SOURCE_PREFIX = "冒烟测试"
SOURCE = f"{SOURCE_PREFIX}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8020")
    args = parser.parse_args()

    failures: list[str] = []

    with httpx.Client(base_url=args.base_url, timeout=60) as client:
        health = client.get("/health").json()
        print(f"健康检查: status={health['status']} chunks={health['chunks']} "
              f"embedder={health['embedder']} llm_configured={health['llm_configured']}")

        indexed = client.post(
            "/knowledge/index", json={"text": SAMPLE, "source": SOURCE, "size": 80}
        ).json()
        print(f"入库: 新增 {indexed['indexed']} 张卡片，库内共 {indexed['total']}（来源 {SOURCE}）")

        search = client.post("/knowledge/search", json={"query": QUESTION, "top_k": 3}).json()
        top = search["hits"][0] if search["hits"] else {}
        print(f"检索首位: score={top.get('score')} 内容={str(top.get('text', ''))[:40]}...")
        if EXPECT_KEYWORD not in str(top.get("text", "")):
            failures.append(f"检索首位未命中「{EXPECT_KEYWORD}」")

        session_id = client.post("/session").json()["session_id"]
        started = time.perf_counter()
        first_token_at = None
        events: list[dict] = []
        with client.stream(
            "POST", f"/session/{session_id}/say", json={"text": QUESTION}
        ) as resp:
            for line in resp.iter_lines():
                if not line.strip():
                    continue
                event = json.loads(line)
                events.append(event)
                if event["type"] == "token" and first_token_at is None:
                    first_token_at = time.perf_counter() - started

        kinds = {e["type"] for e in events}
        for required in ("sources", "token", "done"):
            if required not in kinds:
                failures.append(f"事件流缺少 {required}")
        total = time.perf_counter() - started
        print(f"首字耗时: {(first_token_at or 0) * 1000:.0f} ms；整轮耗时: {total * 1000:.0f} ms")
        print(f"事件类型: {sorted(kinds)}")

        # 清理本次冒烟写入的卡片，保持持久化库干净
        purged = client.post("/knowledge/purge", json={"source_prefix": SOURCE_PREFIX}).json()
        print(f"清理: 删除 {purged['deleted']} 张冒烟卡片，库内剩 {purged['total']}")

    if failures:
        print("\n冒烟未通过：")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("\n冒烟通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
