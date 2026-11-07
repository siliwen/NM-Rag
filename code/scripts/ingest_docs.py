"""批量把非遗资料灌进知识库。

用法（服务需先启动）：
    python scripts/ingest_docs.py ../assets/samples/pdf
    python scripts/ingest_docs.py ../assets/samples/pdf --ocr always --dpi 200
    python scripts/ingest_docs.py 某份扫描件.pdf --max-pages 3      # 先试跑几页
    python scripts/ingest_docs.py 资料目录 --size 400 --overlap 60

支持格式：
    .pdf              上传给服务端解析：有文本层直读，扫描页自动 OCR（见 rag/pdf.py）
    .wav/.mp3/.m4a…   上传给服务端转写：VAD 切句 + 标点 + 时间戳（见 rag/audio.py）
    .txt / .md        直接当文本入库
    .docx / 图片      需要额外依赖，会明确提示缺什么，不静默跳过（NO-006）

音频与 PDF 一样走服务端解析，不在脚本里做：
识别模型只加载一份，判定逻辑也只维护一处，避免脚本和服务端行为漂移。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
PDF_SUFFIXES = {".pdf"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".opus", ".wma", ".amr"}
DOCX_SUFFIXES = {".docx", ".doc"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
# 评测集与标准答案不是资料，灌进去会污染检索
SKIP_SUFFIXES = {".json"}

AUDIO_MIME = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".opus": "audio/opus",
}


def read_text(path: Path) -> str:
    """读纯文本类文件。PDF 不在这里处理——它必须交给服务端解析，
    因为扫描页需要 OCR，而 OCR 的判定逻辑只维护在服务端一处。"""
    if path.suffix.lower() in TEXT_SUFFIXES:
        for encoding in ("utf-8", "utf-8-sig", "gb18030"):
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f"{path.name} 编码无法识别，请转成 UTF-8")

    if path.suffix.lower() in DOCX_SUFFIXES:
        raise RuntimeError(f"{path.name} 是 Word，请先另存为 PDF 再入库（Word 解析未接）")

    if path.suffix.lower() in IMAGE_SUFFIXES:
        raise RuntimeError(
            f"{path.name} 是图片，请先合成为 PDF 再入库（走 OCR 链路）"
        )

    raise RuntimeError(f"{path.name} 格式不支持（{path.suffix}）")


def collect(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    if target.is_dir():
        files = [p for p in sorted(target.rglob("*")) if p.is_file()]
        return [p for p in files if not p.name.startswith(".")]
    raise SystemExit(f"路径不存在: {target}")


def upload_pdf(client: httpx.Client, path: Path, args, rel: Path) -> dict:
    """上传 PDF，交给服务端做文本层/OCR 判定。"""
    with path.open("rb") as fh:
        resp = client.post(
            "/knowledge/index-pdf",
            files={"file": (path.name, fh, "application/pdf")},
            data={
                "source": str(rel).replace("\\", "/"),
                "size": args.size,
                "overlap": args.overlap,
                "ocr": args.ocr,
                "replace": "true" if args.replace else "false",
                "max_pages": args.max_pages,
                "dpi": args.dpi,
                "min_chars": args.min_chars,
            },
            timeout=None,  # OCR 一页要几秒，整本扫描件可能十几分钟
        )
    return resp


def upload_audio(client: httpx.Client, path: Path, args, rel: Path) -> dict:
    """上传录音，交给服务端转写（VAD 切句 + 标点 + 时间戳）。"""
    mime = AUDIO_MIME.get(path.suffix.lower(), "application/octet-stream")
    with path.open("rb") as fh:
        resp = client.post(
            "/knowledge/index-audio",
            files={"file": (path.name, fh, mime)},
            data={
                "source": str(rel).replace("\\", "/"),
                "size": args.size,
                "gap_ms": args.audio_gap_ms,
                "speaker": "true" if args.speaker else "false",
                "max_seconds": args.max_seconds,
                "replace": "true" if args.replace else "false",
            },
            # 一小时录音在 CPU 上要转写几分钟，默认超时会把任务掐断
            timeout=None,
        )
    return resp


def main() -> int:
    parser = argparse.ArgumentParser(description="批量灌入非遗资料")
    parser.add_argument("path", type=Path, help="文件或目录")
    parser.add_argument("--base-url", default="http://127.0.0.1:8020")
    parser.add_argument("--size", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument(
        "--ocr", default="auto", choices=["auto", "always", "never"],
        help="auto=扫描页才 OCR（默认）｜always=全部 OCR｜never=不 OCR",
    )
    parser.add_argument("--dpi", type=int, default=150, help="OCR 渲染精度")
    parser.add_argument("--min-chars", type=int, default=30, help="低于此字符数判为扫描页")
    parser.add_argument("--max-pages", type=int, default=0, help=">0 只处理前 N 页（试跑）")
    parser.add_argument("--audio-gap-ms", type=int, default=1200, help="音频停顿超过该毫秒数即换卡片")
    parser.add_argument("--speaker", action="store_true", help="音频启用说话人分离（更慢）")
    parser.add_argument("--max-seconds", type=int, default=0, help=">0 只转写前 N 秒（长录音试跑）")
    parser.add_argument(
        "--no-replace", action="store_true",
        help="同名来源不覆盖（默认覆盖，避免重复上传后卡片翻倍）",
    )
    args = parser.parse_args()
    args.replace = not args.no_replace

    files = collect(args.path)
    if not files:
        print("没有找到可处理的文件")
        return 1

    ok, failed, total_chunks = 0, 0, 0
    # timeout=None：扫描件 OCR 慢，不能让默认超时把长任务掐断
    with httpx.Client(base_url=args.base_url, timeout=None) as client:
        health = client.get("/health")
        if health.status_code != 200:
            print(f"服务不可用: {args.base_url}/health → {health.status_code}")
            return 1

        for path in files:
            rel = path.relative_to(args.path) if args.path.is_dir() else Path(path.name)
            suffix = path.suffix.lower()

            if suffix in SKIP_SUFFIXES:
                print(f"[跳过] {rel}: 评测/标准答案文件，不入库")
                continue

            if suffix in AUDIO_SUFFIXES:
                try:
                    resp = upload_audio(client, path, args, rel)
                except httpx.HTTPError as exc:
                    print(f"[失败] {rel}: 网络错误 {exc}")
                    failed += 1
                    continue
                if resp.status_code != 200:
                    print(f"[失败] {rel}: HTTP {resp.status_code} {resp.text[:300]}")
                    failed += 1
                    continue
                body = resp.json()
                total_chunks += body["indexed"]
                ok += 1
                extra = ""
                if body["replaced"]:
                    extra += f"，覆盖旧卡片 {body['replaced']} 张"
                print(
                    f"[完成] {rel}: 录音 {body['duration']} → {body['segments']} 句 / "
                    f"新增 {body['indexed']} 张卡片{extra}（库内共 {body['total']}）"
                )
                if body.get("speakers"):
                    print(f"        说话人：{'、'.join(body['speakers'])}")
                for warn in body.get("warnings", []):
                    print(f"        ⚠ {warn}")
                continue

            if suffix in PDF_SUFFIXES:
                try:
                    resp = upload_pdf(client, path, args, rel)
                except httpx.HTTPError as exc:
                    print(f"[失败] {rel}: 网络错误 {exc}")
                    failed += 1
                    continue
                if resp.status_code != 200:
                    detail = resp.text[:300]
                    print(f"[失败] {rel}: HTTP {resp.status_code} {detail}")
                    failed += 1
                    continue
                body = resp.json()
                total_chunks += body["indexed"]
                ok += 1
                extra = ""
                if body["ocr_pages"]:
                    extra += f"，OCR {len(body['ocr_pages'])} 页"
                if body["empty_pages"]:
                    extra += f"，空白 {len(body['empty_pages'])} 页"
                if body["replaced"]:
                    extra += f"，覆盖旧卡片 {body['replaced']} 张"
                print(
                    f"[完成] {rel}: {body['pages']} 页 → 新增 {body['indexed']} 张卡片{extra}"
                    f"（库内共 {body['total']}）"
                )
                for warn in body.get("warnings", []):
                    print(f"        ⚠ {warn}")
                continue

            try:
                text = read_text(path).strip()
            except RuntimeError as exc:
                print(f"[跳过] {rel}: {exc}")
                failed += 1
                continue
            if not text:
                print(f"[跳过] {rel}: 内容为空")
                failed += 1
                continue

            resp = client.post(
                "/knowledge/index",
                json={"text": text, "source": str(rel).replace("\\", "/"), "size": args.size, "overlap": args.overlap},
            )
            if resp.status_code != 200:
                print(f"[失败] {rel}: HTTP {resp.status_code} {resp.text[:120]}")
                failed += 1
                continue
            body = resp.json()
            total_chunks += body["indexed"]
            ok += 1
            print(f"[完成] {rel}: 新增 {body['indexed']} 张卡片（库内共 {body['total']}）")

    print(f"\n汇总：成功 {ok} 个文件，失败 {failed} 个，共入库 {total_chunks} 张卡片")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
