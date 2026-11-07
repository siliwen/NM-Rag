"""PDF 解析测试。

重点不在「能读出文字」，而在那些**不报错但结果错**的地方：
扫描页被判成有字（入库空白卡片）、中文硬换行没合并（切块全碎）、
页码行混进正文、跨页卡片 id 撞车被静默去重。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from feiyi.rag.pdf import (
    OcrUnavailableError,
    PdfDocument,
    PdfPage,
    clean_page_text,
    extract_pdf,
    meaningful_len,
    pdf_chunks,
)

SCAN_HINT = "这一页的文字只能靠 OCR 认出来：陕西华县皮影戏以牛皮刻制影人著称。"


# ------------------------------------------------------------------ 测试替身
class FakeEngine:
    """用文本列表冒充 PDF：text 为空且 has_image=True 的页 = 扫描页。"""

    def __init__(self, pages: list[str], *, images: int | None = None) -> None:
        self._pages = pages
        self.n_pages = len(pages)
        self.rendered: list[int] = []
        self._images = self.n_pages if images is None else images
        self.closed = False

    def page_text(self, index: int) -> str:
        return self._pages[index]

    def render(self, index: int, dpi: int):
        self.rendered.append(index)
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def close(self) -> None:
        self.closed = True


class FakeOcr:
    """每次 read() 依次返回队列里的结果，并记录调用次数。"""

    def __init__(self, *results: list[str]) -> None:
        self._queue = list(results)
        self.calls = 0

    def read(self, image) -> list[str]:
        self.calls += 1
        return self._queue.pop(0) if self._queue else []


# ------------------------------------------------------------------ 有效字符数
def test_meaningful_len_ignores_punctuation_and_space():
    assert meaningful_len(" ——，。\n\t 　 ") == 0
    assert meaningful_len("皮影戏 ABC 123") == 9  # 3 汉字 + 3 字母 + 3 数字


# ------------------------------------------------------------------ 行合并
def test_merges_hard_wrapped_chinese_lines():
    """中文排版换行不是段落结束，必须拼回去，且不留空格。"""
    raw = "为贯彻落实《中华人民共和国国民经济和社会发展第十四\n个五年规划》，进一步加强非遗保护传承工作。"
    out = clean_page_text(raw)
    assert out == "为贯彻落实《中华人民共和国国民经济和社会发展第十四个五年规划》，进一步加强非遗保护传承工作。"
    assert "\n" not in out


def test_keeps_paragraph_break_after_sentence_end():
    raw = "非遗是中华优秀传统文化的重要组成部分。\n保护好、传承好、弘扬好非遗具有重大意义。"
    out = clean_page_text(raw)
    assert out.split("\n") == [
        "非遗是中华优秀传统文化的重要组成部分。",
        "保护好、传承好、弘扬好非遗具有重大意义。",
    ]


def test_block_markers_start_new_paragraph():
    raw = "总体要求如下\n一、指导思想\n坚持系统性保护\n（一）基本原则\n坚持依法保护"
    out = clean_page_text(raw)
    lines = out.split("\n")
    assert "一、指导思想" in lines
    assert "（一）基本原则" in lines


def test_drops_page_number_lines():
    raw = "- 3 -\n正文第一行。\n第 4 页\n共 17 页\n3/17\n正文第二行。"
    out = clean_page_text(raw)
    assert "- 3 -" not in out
    assert "第 4 页" not in out
    assert "共 17 页" not in out
    assert "3/17" not in out
    assert "正文第一行。" in out and "正文第二行。" in out


def test_english_lines_get_space_and_hyphen_joined():
    raw = "This is a long sen-\ntence about shadow puppetry.\nNext line here."
    out = clean_page_text(raw)
    assert "sentence about shadow puppetry." in out
    assert "This is a long sen-" not in out


def test_clean_drops_explicit_lines():
    raw = "文化和旅游部文件\n正文内容。"
    out = clean_page_text(raw, drop_lines=["文化和旅游部文件"])
    assert out == "正文内容。"


# ------------------------------------------------------------------ 抽取主流程
def test_text_layer_page_does_not_trigger_ocr():
    eng = FakeEngine(["正文内容完全够长，不需要 OCR 介入，一共三十个字以上的中文内容填充。"])
    ocr = FakeOcr()
    doc = extract_pdf("x.pdf", engine=eng, ocr_engine=ocr)
    assert ocr.calls == 0
    assert doc.pages[0].method == "text"
    assert doc.ocr_pages == []


def test_blank_page_falls_back_to_ocr():
    eng = FakeEngine([""], images=1)
    ocr = FakeOcr(SCAN_HINT.split("："))
    doc = extract_pdf("扫描件.pdf", engine=eng, ocr_engine=ocr)
    assert ocr.calls == 1
    assert eng.rendered == [0]
    assert doc.pages[0].method == "ocr"
    assert doc.pages[0].used_ocr is True
    assert "陕西华县皮影戏" in doc.pages[0].text
    assert doc.ocr_pages == [1]


def test_ocr_never_marks_page_empty_and_warns():
    eng = FakeEngine([""])
    ocr = FakeOcr(SCAN_HINT.split("："))
    doc = extract_pdf("扫描件.pdf", engine=eng, ocr="never", ocr_engine=ocr)
    assert ocr.calls == 0
    assert doc.pages[0].method == "empty"
    assert doc.pages[0].is_empty
    assert any("跳过 OCR" in w for w in doc.warnings)


def test_ocr_unavailable_raises_instead_of_silently_indexing_blank():
    """OCR 没装时必须炸，不能默默入库一张空卡片（NO-006）。"""

    class BrokenOcr:
        def read(self, image):
            raise OcrUnavailableError("rapidocr 未安装")

    eng = FakeEngine([""])
    with pytest.raises(OcrUnavailableError):
        extract_pdf("扫描件.pdf", engine=eng, ocr_engine=BrokenOcr())


def test_repeated_header_removed_across_pages():
    body = "这一页的正文内容足够长，用来验证页眉去重逻辑是否会误伤正文，一共三十多个汉字。"
    pages = [f"某某县非物质文化遗产申报书\n{body}\n- 1 -"] * 4
    doc = extract_pdf("申报书.pdf", engine=FakeEngine(pages), ocr="never")
    assert "某某县非物质文化遗产申报书" not in doc.pages[0].text
    assert body in doc.pages[0].text
    assert any("重复" in w for w in doc.warnings)


def test_max_pages_limits_work_and_warns():
    pages = ["正文内容足够长的一页，用来测试页数上限是否生效，共三十个汉字以上的内容填充。" for _ in range(5)]
    doc = extract_pdf("大文件.pdf", engine=FakeEngine(pages), max_pages=2)
    assert doc.total_pages == 2
    assert any("前 2/5 页" in w for w in doc.warnings)


def test_invalid_ocr_mode_rejected():
    with pytest.raises(ValueError):
        extract_pdf("x.pdf", engine=FakeEngine(["正文足够长的内容，超过三十个汉字的要求，用于测试参数校验。"]), ocr="sometimes")


def test_engine_closed_only_when_self_created():
    """外部注入的引擎由调用方负责关闭，不能被我顺手关掉。"""
    eng = FakeEngine(["正文足够长的内容，超过三十个汉字的要求，用于测试关闭行为是否正确。"])
    extract_pdf("x.pdf", engine=eng)
    assert eng.closed is False


def test_page_text_extraction_failure_is_explicit():
    class Exploding(FakeEngine):
        def page_text(self, index: int) -> str:
            raise RuntimeError("坏页")

    with pytest.raises(RuntimeError, match="坏页"):
        extract_pdf("坏.pdf", engine=Exploding(["x"]))


# ------------------------------------------------------------------ 切块
def _doc_with_pages(*texts: str) -> PdfDocument:
    return PdfDocument(
        source="申报书.pdf",
        pages=[PdfPage(number=i + 1, text=t, method="text") for i, t in enumerate(texts)],
    )


def test_pdf_chunks_carry_page_metadata():
    doc = _doc_with_pages(
        "皮影戏的影人一般用牛皮或驴皮刻制，经刮薄、雕镂、上色多道工序完成。",
        "华县皮影戏的唱腔以碗碗腔为主，伴奏用月琴与梆子。",
    )
    chunks = pdf_chunks(doc, size=200, overlap=20)
    assert chunks, "应当切出卡片"
    assert {c.meta["page"] for c in chunks} == {1, 2}
    assert all(c.meta["kind"] == "pdf" for c in chunks)
    assert all(c.source == "申报书.pdf" for c in chunks)


def test_pdf_chunks_ids_unique_across_pages():
    """逐页切块若每页从 0 编号，不同页会算出相同 id → 入库静默丢数据。"""
    page = "同样的正文内容重复出现在每一页，用来验证卡片 id 是否会跨页撞车。"
    chunks = pdf_chunks(_doc_with_pages(page, page, page), size=200)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
    assert len(ids) >= 3


def test_pdf_chunks_marks_ocr_page_and_skips_empty():
    doc = PdfDocument(
        source="扫描件.pdf",
        pages=[
            PdfPage(number=1, text="第一页有正常识别的文本内容，长度足够切出一张卡片。", method="text"),
            PdfPage(number=2, text="", method="empty"),
            PdfPage(number=3, text="第三页是 OCR 出来的内容，同样足够长可以切块。", method="ocr"),
        ],
    )
    chunks = pdf_chunks(doc, size=200)
    pages = {c.meta["page"]: c.meta["ocr"] for c in chunks}
    assert pages == {1: False, 3: True}


# ------------------------------------------------------------------ 真实文件
REAL_PDF = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "samples"
    / "pdf"
    / "十四五非物质文化遗产保护规划.pdf"
)


@pytest.mark.skipif(not REAL_PDF.exists(), reason="缺少样例 PDF（assets/samples/pdf）")
def test_real_pdf_extracts_chinese_text():
    doc = extract_pdf(REAL_PDF, ocr="never")
    assert doc.total_pages >= 10
    assert doc.chars > 3000
    assert "非物质文化遗产" in doc.text()
    # 页码与硬换行都要清干净
    first = doc.pages[0].text
    assert "- 1 -" not in first
    assert "\n" in first  # 段落结构保留
    assert len(first.split("\n")[0]) > 8  # 合并后的整段，不是被切碎的一行
