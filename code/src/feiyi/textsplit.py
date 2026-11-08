"""句子切分：流式 TTS 的公共逻辑（编排层与本地 TTS 后端共用，避免各写一套）。

两种用法：
- `split_sentences(text)`           整段文本一次切完 —— 本地 TTS 后端并行合成用
- `pop_complete_sentences(buffer)`  流式切分，吐「已经说完的句子」，剩下的留到下次 —— 编排层按句流式合成用

为什么要切：Kokoro 这类模型一次调用合成一段，整段合成要等全文写完才开始；
切成句子才能「说一句合成一句」，首音延迟从「全文合成」降到「第一句合成」。
"""
from __future__ import annotations

import re

SENT_END = set("。！？；!?;…")
BREAKS = set("，、,：:")
CLOSERS = set("”’）)》」』\"'")

DEFAULT_MAX_CHARS = 40
DEFAULT_MIN_CHARS = 4


def _tidy(text: str) -> str:
    """压掉多余空白：中文不需要词间空格，而大模型常在句间塞两个空格。"""
    return re.sub(r"[ \t\u3000]+", "", text).strip()


def _wrap(text: str, max_chars: int) -> list[str]:
    """超长句在逗号/顿号处硬切，切点之后的逗号留在前一片，保证拼接不丢字。"""
    pieces: list[str] = []
    rest = text
    while len(rest) > max_chars:
        window = rest[:max_chars]
        cut = max((window.rfind(b) for b in BREAKS), default=-1)
        cut = cut + 1 if cut >= max_chars // 2 else max_chars
        pieces.append(rest[:cut])
        rest = rest[cut:]
    if rest:
        pieces.append(rest)
    return pieces


def split_sentences(
    text: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> list[str]:
    """整段切分：按句末标点/换行切，保留标点；过短的合并，过长的在逗号处再切。"""
    raw: list[str] = []
    buf = ""
    i, n = 0, len(text or "")
    while i < n:
        ch = text[i]
        if ch in "\n\r":
            if buf.strip():
                raw.append(_tidy(buf))
            buf = ""
            i += 1
            continue
        buf += ch
        if ch in SENT_END:
            j = i + 1
            # 把紧跟的引号/右括号/重复句末标点一起收进来，避免切出孤零零的标点段
            while j < n and (text[j] in CLOSERS or text[j] in SENT_END):
                buf += text[j]
                j += 1
            raw.append(_tidy(buf))
            buf = ""
            i = j
            continue
        i += 1
    if buf.strip():
        raw.append(_tidy(buf))

    raw = [p for p in raw if p]
    if not raw:
        return []

    merged: list[str] = []
    for part in raw:
        if merged and len(merged[-1]) < min_chars:
            merged[-1] += part
        else:
            merged.append(part)

    out: list[str] = []
    for part in merged:
        out.extend(_wrap(part, max_chars))
    return out


def _window_cut(text: str, max_chars: int, min_chars: int) -> int:
    """在窗口内找最靠右的逗号当切点；实在没有就硬切在窗口边界。

    逗号太靠前（切出来只有几个字）也不划算，那种碎句拼起来发硬，不如按窗口硬切。
    """
    window = text[:max_chars]
    brk = max((window.rfind(b) for b in BREAKS), default=-1) + 1
    return brk if brk >= max(max_chars // 2, min_chars) else max_chars


def _find_boundary(text: str) -> int | None:
    """返回第一个句子边界的切点（含标点/换行）；没有则 None。"""
    for i, ch in enumerate(text):
        if ch in "\n\r":
            return i + 1
        if ch in SENT_END:
            j = i + 1
            while j < len(text) and (text[j] in CLOSERS or text[j] in SENT_END):
                j += 1
            return j
    return None


def pop_complete_sentences(
    buffer: str,
    min_chars: int = DEFAULT_MIN_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
    flush: bool = False,
    allow_break: bool = False,
) -> tuple[list[str], str]:
    """流式切分：(已说完的句子列表, 尚未说完的剩余文本)。

    - 太短（< min_chars）的句子先不吐，把下一句并进来再吐，避免合成出来一截一截的
    - 缓冲区超过 max_chars 还没有句末标点（大模型偶尔一次吐一大段）就在逗号处硬切，
      否则数字人会一直哑着不出声
    - flush=True 表示「流已经结束」：剩下的半句也要吐出来，否则最后一句话永远合不成
    - allow_break=True 允许在逗号处就吐（首段专用）：一句 30 多字要合成 2 秒，
      等句号才切等于白等，先按逗号开口，后面播着的时候再正常切
    """
    out: list[str] = []
    rest = buffer or ""
    while True:
        cut = _find_boundary(rest)
        if cut is None:
            if allow_break:
                brk = max((rest.rfind(b) for b in BREAKS), default=-1) + 1
                if min_chars <= brk <= max_chars:
                    # 窗口内就有逗号：在这儿开口，听觉上最自然
                    out.append(_tidy(rest[:brk]))
                    rest = rest[brk:]
                    continue
                if brk > max_chars or len(rest) > max_chars:
                    # 逗号在窗口之外（或压根没有）：按窗口硬切，别再等整句
                    cut_at = _window_cut(rest, max_chars, min_chars)
                    out.append(_tidy(rest[:cut_at]))
                    rest = rest[cut_at:]
                    continue
            break
        piece = _tidy(rest[:cut])
        if not piece:
            rest = rest[cut:]
            continue
        if cut > max_chars:
            # 整句比窗口还长：先在窗口内切一刀，别为了等整句让数字人哑着
            cut_at = _window_cut(rest, max_chars, min_chars)
            if cut_at < cut:
                out.append(_tidy(rest[:cut_at]))
                rest = rest[cut_at:]
                continue
        if len(piece) < min_chars:
            tail = rest[cut:]
            nxt = _find_boundary(tail)
            if nxt is not None:
                out.append(_tidy(rest[: cut + nxt]))
                rest = rest[cut + nxt :]
                continue
            if not flush:
                break  # 短句后面可能还有内容，先攒着
            out.append(piece)
            rest = tail
            break
        out.append(piece)
        rest = rest[cut:]

    if flush and rest.strip() and len(rest) <= max_chars:
        out.append(_tidy(rest))  # 流已结束：剩下的半句全吐，否则永远没声音
        rest = ""
    elif len(rest) >= max_chars:
        cut = _window_cut(rest, max_chars, min_chars)
        out.append(_tidy(rest[:cut]))
        rest = rest[cut:]

    return [p for p in out if p], rest
