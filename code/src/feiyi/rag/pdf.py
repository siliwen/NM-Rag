"""PDF 解析：文本层优先，扫描页自动降级到 OCR。

为什么不用 PyMuPDF：它是 **AGPL-3.0**，交付给高校等于传染整个交付包。
这里改用 `pypdfium2`（Apache-2.0 + PDFium/BSD），一个库同时满足
「抽文本层」和「渲染成图」两件事，且自带二进制，不需要系统装 poppler。

OCR 用 `rapidocr-onnxruntime`（Apache-2.0），模型随包分发、纯 CPU 可跑，
不联网、不调云、不需要 paddlepaddle（Windows + Python 3.13 上装不动）。

三个必须处理的现实问题（都是「不报错的错」，比崩溃更难查）：
1. **扫描页没有文本层** —— 抽出来是空字符串。直接入库等于这张卡片空白，
   检索永远命中不了，但程序一切正常。所以必须按有效字符数判定并走 OCR。
2. **中文排版硬换行** —— 一行一句地断，不合并就会被切成碎片，
   切块时每行一张卡片，语义全丢。
3. **页码/页眉混进正文** —— 「- 3 -」这类行会被当成句子参与检索打分。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from .chunking import Chunk, chunk_text, make_id

# 判定「这一页到底有没有文字」用的字符集：中文 + 字母 + 数字。
# 不用 len(text)：标点、制表符、软连字符会把空页算成有字。
_MEANING_CHARS = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fffA-Za-z0-9]")

# 纯页码行：- 3 - ／ 3 ／ 第 3 页 ／ 3/17 ／ 共 17 页
_NOISE_LINES = re.compile(
    r"^\s*(?:[-—–－~]?\s*\d{1,4}\s*[-—–－~]?"
    r"|第\s*\d{1,4}\s*页(?:\s*[,，/]?\s*共\s*\d{1,4}\s*页)?"
    r"|\d{1,4}\s*/\s*\d{1,4}"
    r"|共\s*\d{1,4}\s*页)\s*$"
)

# 段落/条目的起始标记：命中就另起一段，不与上一行拼接
_BLOCK_START = re.compile(
    r"^\s*(?:"
    r"[一二三四五六七八九十百]+[、.．]"
    r"|[（(][一二三四五六七八九十百\d]+[)）]"
    r"|\d+[、.．)）]"
    r"|[①-⑳]"
    r"|[-—–·•▪◆◇○●]\s*"
    r"|第[一二三四五六七八九十百\d]+[章节条款项]"
    r"|附(?:件|录)\s*[\d一二三四五六七八九十]*"
    r")"
)

# 句末标点：出现在行尾说明这句说完了，可以断开
_SENT_END = "。！？!?；;…\"”』」）)"

# 上一行行尾的英文连字符（word- 换行断词）
_ASCII_WORD = re.compile(r"[A-Za-z0-9]$")
_ASCII_WORD_START = re.compile(r"^[A-Za-z0-9]")

DEFAULT_MIN_CHARS = 30
DEFAULT_DPI = 150
DUPLICATE_HEADER_RATIO = 0.5
DUPLICATE_HEADER_MAX_LEN = 25


class PdfError(RuntimeError):
    """PDF 处理失败。必须在接口层原样上报，禁止吞掉（NO-006）。"""


class OcrUnavailableError(PdfError):
    """遇到扫描页但 OCR 依赖没装。绝不静默跳过，否则等于入库了一张空白卡片。"""


def meaningful_len(text: str) -> int:
    """有效字符数（中文/字母/数字），用来判定这一页是不是「空的」。"""
    return len(_MEANING_CHARS.findall(text or ""))


# ---------------------------------------------------------------- 引擎接口
class PdfEngine(Protocol):
    """页面来源。抽象出来是为了测试不必准备真实 PDF 文件。"""

    n_pages: int

    def page_text(self, index: int) -> str: ...

    def render(self, index: int, dpi: int) -> Any: ...

    def close(self) -> None: ...


class OcrEngine(Protocol):
    def read(self, image: Any) -> list[str]: ...


class PdfiumEngine:
    """pypdfium2 实现：文本层用 `get_text_range()`，渲染用 `render(scale=)`。"""

    def __init__(self, path: str | Path) -> None:
        try:
            import pypdfium2 as pdfium
        except ImportError as exc:  # pragma: no cover - 环境问题
            raise PdfError("缺少 pypdfium2：pip install pypdfium2") from exc

        self.path = Path(path)
        if not self.path.exists():
            raise PdfError(f"文件不存在：{self.path}")
        try:
            self._doc = pdfium.PdfDocument(str(self.path))
        except Exception as exc:  # pdfium 抛的异常类型不稳定，统一包一层
            raise PdfError(f"无法打开 PDF（可能已损坏或加密）：{self.path.name} → {exc}") from exc
        self.n_pages = len(self._doc)

    def page_text(self, index: int) -> str:
        try:
            return self._doc[index].get_textpage().get_text_range() or ""
        except Exception as exc:  # 单页坏了不该让整份文件失败，但要能被上层看见
            raise PdfError(f"第 {index + 1} 页文本抽取失败：{exc}") from exc

    def render(self, index: int, dpi: int) -> Any:
        """渲染成 BGR ndarray（RapidOCR 的约定输入）。"""
        import numpy as np

        scale = max(0.5, float(dpi) / 72.0)
        try:
            pil = self._doc[index].render(scale=scale).to_pil().convert("RGB")
        except Exception as exc:
            raise PdfError(f"第 {index + 1} 页渲染失败：{exc}") from exc
        return np.array(pil)[:, :, ::-1].copy()

    def close(self) -> None:
        try:
            self._doc.close()
        except Exception:  # pragma: no cover - 关闭失败不影响结果
            pass


class RapidOcrEngine:
    """rapidocr-onnxruntime 实现。懒加载：不碰到扫描页就不加载模型（省 3 秒）。"""

    def __init__(self) -> None:
        self._ocr: Any = None

    def _load(self) -> Any:
        if self._ocr is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
            except ImportError as exc:
                raise OcrUnavailableError(
                    "遇到扫描页需要 OCR，但依赖未安装：pip install rapidocr-onnxruntime"
                ) from exc
            self._ocr = RapidOCR()
        return self._ocr

    def read(self, image: Any) -> list[str]:
        ocr = self._load()
        result, _elapse = ocr(image)
        # 返回结构：[[box, text, score], ...]；按检测框的阅读顺序输出
        return [str(item[1]) for item in (result or []) if len(item) >= 2]


# ---------------------------------------------------------------- 文本清洗
def clean_page_text(raw: str, *, drop_lines: Sequence[str] = ()) -> str:
    """把一页原始文本整理成真正的自然段。

    做四件事：去噪声行 → 去重复页眉 → 合并中文硬换行 → 规整空白。
    """
    text = (raw or "").replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = text.replace("\u00a0", " ").replace("\u3000", " ")

    dropped = {line.strip() for line in drop_lines if line.strip()}
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [
        ln
        for ln in lines
        if ln and not _NOISE_LINES.match(ln) and ln not in dropped
    ]
    if not lines:
        return ""

    merged: list[str] = []
    for line in lines:
        if not merged:
            merged.append(line)
            continue

        prev = merged[-1]
        if _should_break(prev, line):
            merged.append(line)
        else:
            merged[-1] = _join(prev, line)

    return "\n".join(merged).strip()


def _should_break(prev: str, line: str) -> bool:
    """上一行和这一行之间要不要断开。默认不断——中文排版换行不等于段落结束。"""
    if not prev:
        return True
    if prev[-1] in _SENT_END:
        return True
    # 上一行本身就是条目标记（如「一、指导思想」），说明它是标题，下面要另起段
    if _BLOCK_START.match(prev) and len(prev) <= 20:
        return True
    if _BLOCK_START.match(line):
        return True
    # 上一行极短（如小标题）而下行明显是正文，保留标题独立成段
    if len(prev) <= 12 and len(line) > len(prev) * 2:
        return True
    return False


def _join(prev: str, line: str) -> str:
    """拼接两行：中文直接贴，英文词之间补空格，行尾连字符去掉。"""
    if prev.endswith("-") and _ASCII_WORD_START.match(line):
        return prev[:-1] + line
    if _ASCII_WORD.search(prev) and _ASCII_WORD_START.match(line):
        return prev + " " + line
    return prev + line


def _repeated_lines(pages_raw: Sequence[str]) -> set[str]:
    """找出在多数页面重复出现的短行（页眉/页脚/文件名水印）。

    只在页数 ≥ 3 时启用：一两页的文档谈不上「重复」，
    硬删反而可能误伤正常正文。
    """
    if len(pages_raw) < 3:
        return set()
    counter: dict[str, int] = {}
    for raw in pages_raw:
        seen = set()
        for line in (raw or "").split("\n"):
            ln = line.strip()
            if not ln or len(ln) > DUPLICATE_HEADER_MAX_LEN or ln in seen:
                continue
            seen.add(ln)
            counter[ln] = counter.get(ln, 0) + 1
    threshold = max(3, int(len(pages_raw) * DUPLICATE_HEADER_RATIO))
    return {line for line, n in counter.items() if n >= threshold}


# ---------------------------------------------------------------- 结果模型
@dataclass
class PdfPage:
    """一页的抽取结果。"""

    number: int  # 1 起，直接给用户看
    text: str = ""
    method: str = "text"  # text ｜ ocr ｜ empty
    raw_chars: int = 0  # 清洗前有效字符数，用于排查判定是否合理

    @property
    def used_ocr(self) -> bool:
        return self.method == "ocr"

    @property
    def is_empty(self) -> bool:
        return self.method == "empty" or not self.text.strip()


@dataclass
class PdfDocument:
    """整份 PDF 的抽取结果。"""

    source: str
    pages: list[PdfPage] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def total_pages(self) -> int:
        return len(self.pages)

    @property
    def ocr_pages(self) -> list[int]:
        return [p.number for p in self.pages if p.used_ocr]

    @property
    def empty_pages(self) -> list[int]:
        return [p.number for p in self.pages if p.is_empty]

    @property
    def chars(self) -> int:
        return sum(meaningful_len(p.text) for p in self.pages)

    def text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())

    def stats(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "pages": self.total_pages,
            "ocr_pages": self.ocr_pages,
            "empty_pages": self.empty_pages,
            "chars": self.chars,
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------- 主流程
def extract_pdf(
    path: str | Path,
    *,
    source: str = "",
    ocr: str = "auto",
    min_chars: int = DEFAULT_MIN_CHARS,
    dpi: int = DEFAULT_DPI,
    max_pages: int = 0,
    engine: PdfEngine | None = None,
    ocr_engine: OcrEngine | None = None,
    on_page: Callable[[int, int, PdfPage], None] | None = None,
) -> PdfDocument:
    """抽取整份 PDF 的文本，扫描页自动走 OCR。

    参数
    ----
    ocr : "auto" | "always" | "never"
        auto   文本层够用就用文本层，不够的页才 OCR（默认，最快）
        always 所有页都 OCR（用来对比 OCR 与文本层的质量差异）
        never  完全不 OCR，扫描页会被标记为 empty 并给出警告
    min_chars : 一页有效字符数低于此值即判定为扫描页
    dpi : 渲染精度。150 够用；小字/竖排古籍调到 200~300（慢一倍以上）
    max_pages : >0 时只处理前 N 页（试跑大文件用）
    """
    mode = (ocr or "auto").lower()
    if mode not in {"auto", "always", "never"}:
        raise ValueError(f"ocr 只支持 auto/always/never，收到 {ocr!r}")

    own_engine = engine is None
    eng = engine if engine is not None else PdfiumEngine(path)
    name = source or Path(path).name
    doc = PdfDocument(source=name)

    try:
        total = eng.n_pages
        if total <= 0:
            raise PdfError(f"PDF 里没有页面：{name}")
        limit = total if max_pages <= 0 else min(total, max_pages)

        raw_pages: list[str] = []
        pending_ocr: list[int] = []  # 需要 OCR 的页索引，延后统一跑（模型只加载一次）
        raw_texts: dict[int, str] = {}

        for i in range(limit):
            raw = eng.page_text(i)
            raw_texts[i] = raw
            low_text = meaningful_len(raw) < min_chars
            # never 模式也要标记出来：不 OCR 但必须让用户知道这页是空的，
            # 否则扫描件入库后会变成一片空白卡片，而程序毫无反应。
            if mode == "always" or low_text:
                pending_ocr.append(i)
            raw_pages.append(raw)

        repeated = _repeated_lines(raw_pages)
        if repeated:
            doc.warnings.append(
                f"已忽略 {len(repeated)} 条重复出现的页眉/页脚：{'、'.join(sorted(repeated)[:3])}"
                + ("…" if len(repeated) > 3 else "")
            )

        ocr_impl = ocr_engine
        for i in range(limit):
            raw = raw_texts[i]
            use_ocr = i in pending_ocr
            method = "text"
            if use_ocr:
                if mode == "never":
                    method = "empty"
                    doc.warnings.append(
                        f"第 {i + 1} 页没有文本层（疑似扫描页），已按要求跳过 OCR"
                    )
                else:
                    if ocr_impl is None:
                        ocr_impl = RapidOcrEngine()
                    lines = ocr_impl.read(eng.render(i, dpi))
                    raw = "\n".join(lines)
                    method = "ocr"

            text = clean_page_text(raw, drop_lines=repeated)
            page = PdfPage(
                number=i + 1,
                text=text,
                method=method if (method != "text" or text) else "empty",
                raw_chars=meaningful_len(raw),
            )
            if page.method != "empty" and not text.strip():
                page.method = "empty"
            doc.pages.append(page)
            if on_page is not None:
                on_page(i + 1, limit, page)

        if mode != "never" and doc.empty_pages:
            doc.warnings.append(
                f"共 {len(doc.empty_pages)} 页既无文本层也无 OCR 结果："
                f"{doc.empty_pages[:10]}{'…' if len(doc.empty_pages) > 10 else ''}"
                "（可能是纯图片页或空白页）"
            )
        if limit < total:
            doc.warnings.append(f"只处理了前 {limit}/{total} 页（max_pages={max_pages}）")
        return doc
    finally:
        if own_engine:
            eng.close()


def pdf_chunks(
    doc: PdfDocument,
    *,
    size: int = 300,
    overlap: int = 50,
    source: str = "",
) -> list[Chunk]:
    """**逐页**切块，每张卡片带页码。

    为什么不整篇切：整篇切块时页码要靠字符偏移反推，而卡片一旦跨页，
    「出自第几页」就说不清了。逐页切的代价是跨页长段会被切成两块，
    对非遗资料（申报书、文献多半段落密集、页数多）来说，页码可追溯更重要。

    卡片 id 用全局递增序号生成：若按页各自从 0 开始，不同页的同位置卡片
    会算出同一个 id，后入库的会被 store 当成重复直接丢掉——静默丢数据。
    """
    src = source or doc.source
    out: list[Chunk] = []
    for page in doc.pages:
        if not page.text.strip():
            continue
        for chunk in chunk_text(page.text, size=size, overlap=overlap, source=src):
            chunk.id = make_id(src, len(out), chunk.text)
            chunk.meta = {
                "kind": "pdf",
                "page": page.number,
                "ocr": page.used_ocr,
            }
            out.append(chunk)
    return out


def extract_pdf_chunks(
    path: str | Path,
    *,
    source: str = "",
    size: int = 300,
    overlap: int = 50,
    ocr: str = "auto",
    min_chars: int = DEFAULT_MIN_CHARS,
    dpi: int = DEFAULT_DPI,
    max_pages: int = 0,
) -> tuple[PdfDocument, list[Chunk]]:
    """一步到位：抽取 + 切块。CLI 和服务端都走这里，避免两套行为漂移。"""
    doc = extract_pdf(
        path,
        source=source,
        ocr=ocr,
        min_chars=min_chars,
        dpi=dpi,
        max_pages=max_pages,
    )
    return doc, pdf_chunks(doc, size=size, overlap=overlap, source=source or doc.source)
