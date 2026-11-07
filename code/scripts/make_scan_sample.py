"""生成「模拟扫描件」样本：把文字版 PDF 的若干页渲染成图片，再打包成不含文本层的 PDF。

为什么需要它：验证 OCR 链路的真实性。网上能找到的非遗 PDF 绝大多数带文本层，
把它们直接喂给解析器，走的是文本层分支，OCR 那半条路根本没被执行过——
"测试通过"是假的。这里把页面栅格化成纯图像 PDF（可选加噪点与倾斜），
强行制造"没有文本层"的条件。

用法：
    python scripts/make_scan_sample.py assets/samples/pdf/十四五非物质文化遗产保护规划.pdf \
        --pages 1-3 --out assets/samples/pdf/模拟扫描件-非遗规划.pdf --noise 6 --rotate 0.6
"""
from __future__ import annotations

import argparse
import io
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_pages(spec: str, total: int) -> list[int]:
    """'1-3,5' → [1,2,3,5]（1 起，转成 0 起的索引）。"""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    picked = [p - 1 for p in out if 1 <= p <= total]
    if not picked:
        raise SystemExit(f"页码 {spec} 超出范围（共 {total} 页）")
    return picked


def main() -> int:
    ap = argparse.ArgumentParser(description="把文字版 PDF 转成模拟扫描件")
    ap.add_argument("source", type=Path)
    ap.add_argument("--pages", default="1-2", help="页码范围，如 1-3,5")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--noise", type=int, default=0, help="叠加噪点强度（0 不加噪）")
    ap.add_argument("--rotate", type=float, default=0.0, help="轻微倾斜角度，模拟扫描歪斜")
    args = ap.parse_args()

    import numpy as np
    import pypdfium2 as pdfium
    from PIL import Image

    if not args.source.exists():
        raise SystemExit(f"源文件不存在：{args.source}")

    doc = pdfium.PdfDocument(str(args.source))
    idx = parse_pages(args.pages, len(doc))
    scale = args.dpi / 72.0
    rng = random.Random(20261105)  # 固定种子，样本可重复生成

    images: list[Image.Image] = []
    for i in idx:
        img = doc[i].render(scale=scale).to_pil().convert("L")  # 灰度更像扫描件
        if args.rotate:
            img = img.rotate(args.rotate, resample=Image.BICUBIC, fillcolor=255, expand=False)
        if args.noise:
            arr = np.array(img).astype(np.int16)
            arr += np.random.default_rng(rng.randint(0, 10**6)).normal(
                0, args.noise, arr.shape
            ).astype(np.int16)
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        images.append(img)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    images[0].save(buf, format="PDF", save_all=True, append_images=images[1:], resolution=args.dpi)
    args.out.write_bytes(buf.getvalue())

    size_kb = args.out.stat().st_size / 1024
    print(f"已生成 {args.out}（{len(images)} 页，{size_kb:.0f} KB，无文本层）")

    # 立刻自检：确认产出的 PDF 真的抽不出文字，否则"模拟扫描件"名不副实
    out_doc = pdfium.PdfDocument(str(args.out))
    chars = sum(len((out_doc[i].get_textpage().get_text_range() or "").strip()) for i in range(len(out_doc)))
    print(f"自检：新文件文本层字符数 = {chars}（应为 0，否则不是真正的扫描件）")
    return 0 if chars == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
