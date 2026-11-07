"""口述史转写质量评测：字准率（CER）+ 时间戳误差 + 切块边界合规。

有了标准答案才能谈质量。这个脚本对着 `make_oral_history_sample.py`
产出的 `.truth.json` 逐项核对，输出可以直接写进交付报告的数字。

三项指标各自回答一个不同的问题：
- **CER**：认字认得准不准（1 - 字错率，越高越好）
- **时间戳误差**：用户点「03:21」跳过去，是不是真的听到那句话
- **边界违规**：有没有把两段不同话题粘进同一张卡片
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from feiyi.rag.audio import (  # noqa: E402
    FunAsrEngine,
    Segment,
    Transcript,
    fmt_time,
    transcribe_audio,
    transcript_chunks,
)

# 只保留中文/字母/数字：标点和空格不该算错字
def _norm(text: str) -> str:
    return "".join(ch for ch in text if ch.isalnum())


def cer(ref: str, hyp: str) -> float:
    """字错率 = 编辑距离 / 参考字数。用两行滚动数组，长文本也不吃内存。"""
    r, h = _norm(ref), _norm(hyp)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        cur = [i] + [0] * len(h)
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[len(h)] / len(r)


def _overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def timestamp_errors(truth: list[dict], preds: list[Segment]) -> list[dict]:
    """为每句标准答案找「覆盖这句最早位置」的识别段，算起点偏差。

    不能取「时间重叠最多」的那段：识别结果会按逗号把一句话拆成好几段，
    重叠最多的往往是最长的那个分句（常在句尾），量出来的是句内长度，
    不是时间戳误差。取最早开始的那段，才真正回答「这句从哪一刻开始」。
    """
    out: list[dict] = []
    for t in truth:
        t_start, t_end = t["start_ms"], t["end_ms"]
        hits = [p for p in preds if _overlap(t_start, t_end, p.start_ms, p.end_ms) > 0]
        best = min(hits, key=lambda p: p.start_ms) if hits else None
        out.append(
            {
                "truth": t["text"],
                "truth_start": t_start,
                "pred": best.text if best else None,
                "pred_start": best.start_ms if best else None,
                "error_ms": abs(best.start_ms - t_start) if best else None,
                "matched": bool(best),
            }
        )
    return out


def boundary_violations(chunks, truth: list[dict]) -> list[dict]:
    """检查有没有卡片跨过「段落之间的长停顿」。

    段落边界在真值里等于一次 1.5 秒静音。一张卡片如果起点在前一段、
    终点在后一段，说明切块没有在停顿处断开——两个话题被粘在一起，
    检索时会互相污染。
    """
    breaks: list[int] = []
    for prev, cur in zip(truth, truth[1:]):
        if cur["paragraph"] != prev["paragraph"]:
            breaks.append((prev["end_ms"] + cur["start_ms"]) // 2)
    bad: list[dict] = []
    for chunk in chunks:
        start = chunk.meta["start_ms"]
        end = chunk.meta["end_ms"]
        for point in breaks:
            if start < point < end:
                bad.append(
                    {
                        "chunk": chunk.text[:30],
                        "start": fmt_time(start),
                        "end": fmt_time(end),
                        "gap_at": fmt_time(point),
                    }
                )
    return bad


def _pct(values: list[int], q: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    idx = min(len(values) - 1, int(round(q * (len(values) - 1))))
    return values[idx]


def main() -> int:
    ap = argparse.ArgumentParser(description="口述史转写质量评测")
    ap.add_argument("audio")
    ap.add_argument("--truth", default="", help="标准答案 json，默认取同名 .truth.json")
    ap.add_argument("--json", default="", help="把结果写到指定 json")
    ap.add_argument("--speaker", action="store_true", help="启用说话人分离（更慢）")
    ap.add_argument("--dump", default="", help="把识别段存成 json，供后续离线分析/复用")
    ap.add_argument("--segments", default="", help="直接读已存的识别段，跳过（很慢的）识别")
    ap.add_argument("--gap-ms", type=int, default=1200, help="切块停顿阈值，用于边界检查")
    args = ap.parse_args()

    audio = Path(args.audio)
    truth_path = Path(args.truth) if args.truth else audio.with_suffix(".truth.json")
    if not truth_path.exists():
        raise SystemExit(f"找不到标准答案：{truth_path}（先用 make_oral_history_sample.py 生成）")
    truth_doc = json.loads(truth_path.read_text(encoding="utf-8"))
    truth = truth_doc["sentences"]

    if args.segments:
        # 复用已存的识别结果：调阈值、改指标时不必每次重跑识别
        raw = json.loads(Path(args.segments).read_text(encoding="utf-8"))
        tr = Transcript(
            source=raw.get("source", audio.name),
            duration_ms=raw.get("duration_ms", 0),
            segments=[Segment(**s) for s in raw["segments"]],
            engine="cached",
        )
        elapsed = 0.0
    else:
        engine = FunAsrEngine(spk_model="cam++" if args.speaker else "")
        # 先把模型加载单独计时：混在一起会把 RTF 报成 0.4（实际 0.09），
        # 使用者会据此误判"CPU 跑不动口述史"。
        t0 = time.time()
        engine.load()
        load_s = time.time() - t0
        t0 = time.time()
        tr = transcribe_audio(audio, engine=engine)
        elapsed = time.time() - t0
        print(f"模型加载 {load_s:.1f}s")

    if args.dump:
        Path(args.dump).write_text(
            json.dumps(
                {
                    "source": tr.source,
                    "duration_ms": tr.duration_ms,
                    "segments": [
                        {"start_ms": s.start_ms, "end_ms": s.end_ms, "text": s.text, "speaker": s.speaker}
                        for s in tr.segments
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    chunks = transcript_chunks(tr, gap_ms=args.gap_ms)

    ref_text = "".join(t["text"] for t in truth)
    err = cer(ref_text, tr.text())
    ts = timestamp_errors(truth, tr.segments)
    matched = [t["error_ms"] for t in ts if t["matched"]]
    bad = boundary_violations(chunks, truth)

    audio_s = truth_doc["duration_ms"] / 1000
    report = {
        "audio": audio.name,
        "duration_s": round(audio_s, 1),
        "asr_seconds": round(elapsed, 1),
        "rtf": round(elapsed / audio_s, 3) if (audio_s and elapsed) else None,
        "cer": round(err, 4),
        "accuracy": round(1 - err, 4),
        "ref_chars": len(_norm(ref_text)),
        "hyp_chars": len(_norm(tr.text())),
        "segments": len(tr.segments),
        "truth_sentences": len(truth),
        "matched": len(matched),
        "ts_error_ms": {
            "median": _pct(matched, 0.5),
            "p90": _pct(matched, 0.9),
            "max": max(matched) if matched else 0,
        },
        "chunks": len(chunks),
        "boundary_violations": bad,
        "samples": [{"truth": t["truth"], "pred": t["pred"]} for t in ts][:8],
    }

    print(f"\n=== {audio.name} ===")
    timing = f"转写耗时 {elapsed:.1f}s ｜ RTF {report['rtf']:.3f}" if elapsed else "（复用已存识别结果）"
    print(f"时长 {audio_s:.1f}s ｜ {timing}")
    print(f"字准率 {report['accuracy'] * 100:.2f}%  (CER {err * 100:.2f}%, 参考 {report['ref_chars']} 字)")
    print(f"识别段 {len(tr.segments)} / 真值句 {len(truth)} ｜ 时间戳误差 中位 {report['ts_error_ms']['median']}ms ｜ P90 {report['ts_error_ms']['p90']}ms ｜ 最大 {report['ts_error_ms']['max']}ms")
    print(f"切块 {len(chunks)} 张 ｜ 跨长停顿违规 {len(bad)} 处")
    if bad:
        for b in bad[:5]:
            print(f"   ✗ {b['start']}-{b['end']} 跨过 {b['gap_at']} 的停顿：{b['chunk']}")
    print("对照抽样：")
    for s in report["samples"][:5]:
        print(f"   真值：{s['truth']}")
        print(f"   识别：{s['pred']}")

    if args.json:
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果已写入 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
