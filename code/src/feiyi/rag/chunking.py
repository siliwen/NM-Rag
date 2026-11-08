"""文本切块：把长篇非遗资料切成一张张可检索的「卡片」。"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;])|\n+")

# 标题行里的「起止时间」，例如 2020.05-2020.12 / 2016.05—2018.04 / 2024.07至2025.03
# 只认区间，不认单个年份：正文里「2023年，负责…」很常见，按单年份判断会把正文切碎
_HEADING_PERIOD = re.compile(
    r"(?:19|20)\d{2}\s*[.\-/年]?\s*\d{0,2}\s*[-–—~至]\s*(?:19|20)\d{2}"
)


def looks_like_heading(piece: str, max_chars: int = 60) -> bool:
    """判断一段是否像「标题行」：短、不以句末标点收尾、含起止时间。

    为什么要单独识别：简历/履历里「项目名 + 角色 + 起止时间」是标题行，
    如果被并进上一张卡的末尾，后面描述项目的卡片就**没有项目名**了。
    检索到那种卡时，大模型只能自己编一个项目名——这是最坏的一类错误：
    程序不报错，输出却是假的。
    """
    p = (piece or "").strip()
    if not p or len(p) > max_chars:
        return False
    if p.endswith(("。", "；", ";", "，", ",", "：", ":")):
        return False
    return bool(_HEADING_PERIOD.search(p))


def split_title_spans(text: str, max_head_chars: int = 60) -> list[tuple[str, bool]]:
    """把一段文本切成 [(片段, 是否标题)]，标题 =「名称 + 角色 + 起止时间」。

    为什么要做到「段内切分」：PDF 提取经常丢换行，标题后面也没有标点，
    于是标题和描述黏成一段（「…2021.09-2022.01通，事件上报链路长…」）。
    只按换行/句末标点判断的话，这种标题永远识别不出来，项目名就会丢。

    做法：先定位起止时间，再向左找最近的句末标点，中间那段就是标题。
    """
    out: list[tuple[str, bool]] = []
    pos = 0
    for m in _HEADING_PERIOD.finditer(text):
        end = m.end()
        start = pos
        # 标题起点：从时间区间结尾向前找最近的句末标点，最多回看 80 字
        for i in range(end - 1, max(pos, end - 80) - 1, -1):
            if text[i] in "。！？；;\n":
                start = i + 1
                break
        head = text[start:end].strip()
        if not head or len(head) > max_head_chars:
            continue  # 不像标题（区间前面是一大段话），跳过这个时间
        if start > pos:
            out.append((text[pos:start], False))
        out.append((head, True))
        pos = end
    if pos < len(text):
        out.append((text[pos:], False))
    return out


@dataclass
class Chunk:
    """一张卡片。"""

    id: str
    text: str
    source: str = ""
    meta: dict = field(default_factory=dict)


def make_id(source: str, index: int, text: str) -> str:
    """卡片 id：来源 + 序号 + 正文前 40 字。

    公开导出，是因为 PDF 解析要逐页切块后**全局重排序号**：
    每页各自从 0 开始时，不同页的同位置卡片会算出相同 id，
    入库时被当成重复静默丢弃。
    """
    raw = f"{source}|{index}|{text[:40]}".encode("utf-8")
    return hashlib.blake2b(raw, digest_size=8).hexdigest()


# 旧名保留，避免外部脚本引用失效
_make_id = make_id


def split_sentences(text: str) -> list[str]:
    """按中文句末标点与换行切句，保留标点。"""
    parts = _SENT_SPLIT.split(text or "")
    return [p.strip() for p in parts if p and p.strip()]


def _hard_split(sentence: str, size: int) -> list[str]:
    return [sentence[i : i + size] for i in range(0, len(sentence), size)]


def chunk_text(
    text: str,
    size: int = 300,
    overlap: int = 50,
    source: str = "",
    min_size: int = 20,
) -> list[Chunk]:
    """切块。

    - size: 单块目标长度（字符）
    - overlap: 相邻块的重叠字符数，避免句子被切断后语义丢失
    - min_size: 过短的尾块直接丢弃（除非全文只有这一块）
    """
    text = (text or "").strip()
    if not text:
        return []

    overlap = max(0, min(overlap, size // 2))

    # 先按句切，再把每句里的「标题」剥成独立片段（PDF 丢换行时标题会和描述黏在一起）
    spans: list[tuple[str, bool]] = []
    for sent in split_sentences(text):
        for seg, is_heading in split_title_spans(sent):
            seg = seg.strip()
            if not seg:
                continue
            if not is_heading and len(seg) > size:
                spans.extend((piece, False) for piece in _hard_split(seg, size))
            else:
                spans.append((seg, is_heading))

    chunks: list[Chunk] = []
    buf = ""
    for piece, heading in spans:
        if buf and (heading or len(buf) + len(piece) > size):
            chunks.append(Chunk(id=_make_id(source, len(chunks), buf), text=buf.strip(), source=source))
            # 标题开新块时不带上一块的尾巴：否则块首混进无关内容，检索会串味
            buf = "" if heading else (buf[-overlap:] if overlap else "")
        buf += piece

    tail = buf.strip()
    if tail and (len(tail) >= min_size or not chunks):
        chunks.append(Chunk(id=_make_id(source, len(chunks), tail), text=tail, source=source))

    return chunks
