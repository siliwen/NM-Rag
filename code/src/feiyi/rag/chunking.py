"""文本切块：把长篇非遗资料切成一张张可检索的「卡片」。"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;])|\n+")


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

    pieces: list[str] = []
    for sent in split_sentences(text):
        if len(sent) > size:
            pieces.extend(_hard_split(sent, size))
        else:
            pieces.append(sent)

    chunks: list[Chunk] = []
    buf = ""
    for piece in pieces:
        if buf and len(buf) + len(piece) > size:
            chunks.append(Chunk(id=_make_id(source, len(chunks), buf), text=buf.strip(), source=source))
            buf = buf[-overlap:] if overlap else ""
        buf += piece

    tail = buf.strip()
    if tail and (len(tail) >= min_size or not chunks):
        chunks.append(Chunk(id=_make_id(source, len(chunks), tail), text=tail, source=source))

    return chunks
