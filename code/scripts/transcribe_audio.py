"""口述史转写工作台：转写 → 导出可编辑稿 → 人工校对 → 回灌入库。

为什么要有这个脚本，而不是让服务端一把梭：
传承人讲的是方言，`paraformer-zh` 是普通话模型，识别率不会高到能直接用的程度。
正确的流程是「机器转写 + 人工改错字」，而改错字必须在**带时间戳的纯文本**上做，
改完还要能把时间戳读回来。这就是本脚本存在的理由。

四种用法：

    # 1. 只转写，导出可编辑稿（不需要启动服务）
    python scripts/transcribe_audio.py 王守义口述史.wav

    # 2. 转写并直接入库（音频上传给服务端，服务端跑识别）
    python scripts/transcribe_audio.py 王守义口述史.wav --ingest

    # 3. 校对完的稿子回灌（服务端完全不跑识别，也不需要模型）
    python scripts/transcribe_audio.py --from-transcript 王守义口述史.transcript.txt --ingest

    # 4. 长录音先试跑前 3 分钟
    python scripts/transcribe_audio.py 访谈.wav --max-seconds 180

为什么不直接把音频传给服务端入库：一小时录音在 CPU 上要转写十几分钟，
放在 HTTP 请求里既容易超时，也没法边转边看。脚本模式可以先看稿子再入库。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import httpx  # noqa: E402

from feiyi.rag.audio import (  # noqa: E402
    FunAsrEngine,
    fmt_time,
    format_transcript,
    load_transcript,
    save_transcript,
    transcribe_audio,
    transcript_chunks,
)


def transcribe_locally(path: Path, args) -> tuple[object, float, float]:
    """返回 (transcript, 加载耗时, 转写耗时)。两者分开报，否则 RTF 会被加载时间污染。"""
    engine = FunAsrEngine(spk_model="cam++" if args.speaker else "")
    started = time.time()
    engine.load()
    load_seconds = time.time() - started
    started = time.time()
    tr = transcribe_audio(path, engine=engine, max_seconds=args.max_seconds)
    return tr, load_seconds, time.time() - started


def ingest_via_api(text: str, source: str, args) -> int:
    """把（校对后的）转写稿交给服务端入库。服务端不需要 ASR。"""
    payload = {
        "source": source,
        "size": args.size,
        "gap_ms": args.gap_ms,
        "replace": "true" if args.replace else "false",
        "transcript": text,
    }
    with httpx.Client(base_url=args.base_url, timeout=None) as client:
        health = client.get("/health")
        if health.status_code != 200:
            print(f"服务不可用：{args.base_url}/health → {health.status_code}")
            return 1
        resp = client.post(
            "/knowledge/index-audio",
            files={"file": (f"{source}.txt", b"", "text/plain")},
            data=payload,
        )
    if resp.status_code != 200:
        print(f"[失败] 入库返回 {resp.status_code}：{resp.text[:400]}")
        return 2
    body = resp.json()
    print(
        f"[入库] {body['source']}：{body['segments']} 段 → {body['indexed']} 张卡片"
        f"（库内共 {body['total']}）"
        + (f"，覆盖旧卡片 {body['replaced']} 张" if body["replaced"] else "")
    )
    for warn in body.get("warnings", []):
        print(f"        ⚠ {warn}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="口述史转写与校对稿回灌")
    ap.add_argument("audio", nargs="?", help="音频文件（用 --from-transcript 时可省略）")
    ap.add_argument("--from-transcript", default="", help="读取已校对好的转写稿，不再跑识别")
    ap.add_argument("--out", default="", help="校对稿输出路径，默认 <音频名>.transcript.txt")
    ap.add_argument("--jsonl", action="store_true", help="同时导出结构化 jsonl")
    ap.add_argument("--ingest", action="store_true", help="入库（走 HTTP 接口）")
    ap.add_argument("--base-url", default="http://127.0.0.1:8020")
    ap.add_argument("--size", type=int, default=300, help="单张卡片字数")
    ap.add_argument("--gap-ms", type=int, default=1200, help="停顿超过该毫秒数即换卡片")
    ap.add_argument("--speaker", action="store_true", help="启用说话人分离（更慢）")
    ap.add_argument("--max-seconds", type=int, default=0, help=">0 只转写前 N 秒（试跑）")
    ap.add_argument("--no-replace", action="store_true", help="同名来源不覆盖旧卡片")
    args = ap.parse_args()
    args.replace = not args.no_replace

    if not args.from_transcript and not args.audio:
        ap.error("要么给音频文件，要么用 --from-transcript 指定校对稿")

    if args.from_transcript:
        source = Path(args.from_transcript)
        tr = load_transcript(source)
        if not tr.segments:
            print(f"[失败] {source} 里没有解析出任何句子")
            return 2
        stem = source.name.replace(".transcript.txt", "").replace(".txt", "").replace(".jsonl", "")
        name = stem
        print(f"[读取] {source.name}：{len(tr.segments)} 段，时长 {tr.stats()['duration']}")
        text = format_transcript(tr, header=True)
    else:
        audio = Path(args.audio)
        if not audio.exists():
            print(f"[失败] 音频不存在：{audio}")
            return 1
        name = audio.name
        print(f"[转写] {audio.name} …（首次运行要下载模型，约 900MB）")
        tr, load_s, asr_s = transcribe_locally(audio, args)
        stats = tr.stats()
        rtf = asr_s / (stats["duration_ms"] / 1000) if stats["duration_ms"] else 0
        print(
            f"[完成] {stats['duration']} ｜ {stats['segments']} 句 ｜ {stats['chars']} 字 ｜ "
            f"模型加载 {load_s:.1f}s + 转写 {asr_s:.1f}s（RTF {rtf:.3f}）"
        )
        for warn in stats["warnings"]:
            print(f"        ⚠ {warn}")
        if not tr.segments:
            print("[中止] 没有识别出内容，不生成空校对稿")
            return 2

        out = Path(args.out) if args.out else audio.with_suffix(".transcript.txt")
        save_transcript(tr, out)
        print(f"[导出] {out}")
        if args.jsonl:
            print(f"[导出] {save_transcript(tr, out.with_suffix('.transcript.jsonl'))}")
        text = format_transcript(tr, header=True)

    chunks = transcript_chunks(tr, size=args.size, gap_ms=args.gap_ms, source=name)
    print(f"[切块] {len(chunks)} 张卡片，样例：")
    for c in chunks[:3]:
        print(f"        {fmt_time(c.meta['start_ms'])}-{fmt_time(c.meta['end_ms'])}  {c.text[:36]}…")

    if args.ingest:
        return ingest_via_api(text, name, args)

    if not args.from_transcript:
        print("\n下一步：用编辑器打开上面那份 txt 改错字（改完删掉 # 开头的注释行也可以），"
              "然后执行：\n  python scripts/transcribe_audio.py --from-transcript <校对稿> --ingest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
