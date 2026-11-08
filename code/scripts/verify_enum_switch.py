"""回归验证：清单模式 vs 普通问答的分流是否都正确。

关注三件事：
1. 「列出所有项目」仍走清单模式（全量取卡、时间倒序、不合成语音）
2. 「秦腔有哪些角色行当」虽含列举词，但只命中少数卡，必须回到普通问答并且有语音
3. 普通问答首音延迟没被这次改动拖慢

用法：python scripts/verify_enum_switch.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8020"

QUESTIONS = [
    "把简历里所有项目列一个清单出来",
    "秦腔有哪些角色行当？",
    "皮影戏是怎么做的？",
]


def run(client: httpx.Client, question: str) -> dict:
    resp = client.post(f"{BASE}/session")
    resp.raise_for_status()
    sid = resp.json()["session_id"]

    started = time.perf_counter()
    first_audio = None
    sources = 0
    audio_seqs = []
    tokens = []
    with client.stream("POST", f"{BASE}/session/{sid}/say", json={"text": question}) as stream:
        for line in stream.iter_lines():
            if not line.strip():
                continue
            event = json.loads(line)
            kind = event.get("type")
            if kind == "sources":
                sources = len(event.get("items", []))
            elif kind == "token":
                tokens.append(event.get("text", ""))
            elif kind == "audio":
                if first_audio is None:
                    first_audio = time.perf_counter() - started
                audio_seqs.append(event.get("seq"))
    total = time.perf_counter() - started
    text = "".join(tokens)
    return {
        "question": question,
        "sources": sources,
        "audio": len(audio_seqs),
        "first_audio": first_audio,
        "chars": len(text),
        "total": total,
        "text": text,
    }


def main() -> int:
    client = httpx.Client(timeout=180.0, trust_env=False)
    for question in QUESTIONS:
        r = run(client, question)
        first = f"{r['first_audio']:.2f}s" if r["first_audio"] is not None else "无音频"
        print(f"\n【{r['question']}】")
        print(
            f"  召回 {r['sources']} 张 | 音频 {r['audio']} 段 | 首音 {first} "
            f"| 回答 {r['chars']} 字 | 总耗时 {r['total']:.2f}s"
        )
        print(f"  开头：{r['text'][:120]}")
    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
