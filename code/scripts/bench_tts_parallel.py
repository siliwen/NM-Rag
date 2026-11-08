"""对比「整段串行合成」与「切句并行合成」的耗时，并给出并行度建议。

用法：
    python scripts/bench_tts_parallel.py
    python scripts/bench_tts_parallel.py --workers 1 2 4 8
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import kokoro_tts_server as tts  # noqa: E402

TEXT = (
    "皮影戏就是用灯光把皮做的影人照在幕布上讲故事，也叫灯影戏。"
    "老祖宗汉武帝时就有人用灯光照影子招魂，这算是最早的雏形了。"
    "现在演戏时，艺人躲在幕后操纵皮影，配上唱腔和锣鼓，热闹得很。"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", default="zf_001")
    parser.add_argument("--speed", type=float, default=tts.SPEED)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    args = parser.parse_args()

    pipeline = tts.get_pipeline()
    voice = tts.resolve_voice(args.voice)
    parts = tts.split_sentences(TEXT)
    print(f"文本 {len(TEXT)} 字 -> 切 {len(parts)} 片，语速 {args.speed}")
    for p in parts:
        print(f"  · {p}")

    base = None
    for w in args.workers:
        start = time.perf_counter()
        audio, _ = tts.synth_all(pipeline, TEXT, voice=voice, speed=args.speed, workers=w)
        cost = time.perf_counter() - start
        dur = len(audio) / tts.SAMPLE_RATE
        base = base or cost
        print(
            f"workers={w}: 合成 {cost:.2f}s / 音频 {dur:.2f}s -> RTF {cost / dur:.3f}"
            f" | 相对串行 {cost / base * 100:.0f}%"
        )


if __name__ == "__main__":
    main()
