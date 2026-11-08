"""量一次完整问答里各环节的到达时刻，判断「慢」是等太久还是语速慢。

按句流式改造后，音频会分多段到达：这里逐段打印到达时刻与句子内容，
首音延迟看第一段的到达时间。
"""
import base64
import io
import json
import time

import httpx
import numpy as np
import soundfile as sf

BASE = "http://127.0.0.1:8020"
QUESTION = "皮影戏是什么？"

sid = httpx.post(f"{BASE}/session", trust_env=False).json()["session_id"]
t0 = time.perf_counter()
marks: dict[str, float] = {}
tokens: list[str] = []
segments: list[tuple[float, dict]] = []

with httpx.stream("POST", f"{BASE}/session/{sid}/say", json={"text": QUESTION}, timeout=180, trust_env=False) as r:
    for line in r.iter_lines():
        if not line.strip():
            continue
        ev = json.loads(line)
        now = time.perf_counter() - t0
        marks.setdefault(ev["type"], now)
        if ev["type"] == "token":
            tokens.append(ev["text"])
            marks["last_token"] = now
        elif ev["type"] == "audio":
            marks.setdefault("audio", now)
            segments.append((now, ev))
        elif ev["type"] == "error":
            print(f"[error] {ev.get('state')}")

text = "".join(tokens)
print(f"问题：{QUESTION}")
print(f"首 token 文本 : {marks.get('token', 0):.2f}s")
print(f"末 token      : {marks.get('last_token', 0):.2f}s")
print(f"首音（第 1 段）: {marks.get('audio', 0):.2f}s   <-- 首音延迟")
print(f"done          : {marks.get('done', 0):.2f}s")
print(f"答案字数：{len(text)}，音频段数：{len(segments)}")
print("--- 分段到达 ---")

total = 0.0
for idx, (at, ev) in enumerate(segments, start=1):
    raw = base64.b64decode(ev["data"])
    d, sr = sf.read(io.BytesIO(raw), dtype="float32")
    dur = len(d) / sr
    total += dur
    cost = ev.get("elapsed_ms")
    cost_s = f"合成 {cost / 1000:4.2f}s" if cost else "合成     ?"
    print(f"  #{ev.get('seq', idx)} @{at:5.2f}s  音频 {dur:5.2f}s  {cost_s}  {ev.get('text', '')[:22]}")

if segments:
    first_at = segments[0][0]
    wait = marks.get("last_token", 0) - first_at
    print(f"\n首段早于文字结束 {wait:+.2f}s（正数=文字还没出完就已经开口）")
    print(f"音频总时长 {total:.2f}s，整轮 {marks.get('done', 0):.2f}s")
    print(f"语速：{len(text) / total:.2f} 字/秒（正常中文播报约 4~5 字/秒）")
