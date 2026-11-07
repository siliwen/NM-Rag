"""向量库：内存实现（测试/临时）与 SQLite 持久化实现（生产默认）。

为什么不用 Chroma / Milvus：
非遗知识库规模在几千到几万张卡片，这个量级的暴力余弦检索是毫秒级的
（numpy 矩阵乘一次算完），引入向量数据库要额外背上 onnxruntime / tokenizers
等上百 MB 依赖，交付包和运维面都变大。`VectorStore` 接口是稳定的，
真到百万级时换 Milvus 只需新写一个实现，编排层不动。

SQLite 里向量按 float32 BLOB 存，加载时拼成一个矩阵并预归一化，
检索就是一次 `矩阵 @ 查询向量`，比逐条计算快两个数量级。
"""
from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from .chunking import Chunk


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    detail: dict = field(default_factory=dict)


class VectorStore(Protocol):
    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None: ...

    def search(self, vector: Sequence[float], top_k: int) -> list[ScoredChunk]: ...

    def chunks(self) -> list[Chunk]: ...

    def delete_by_source_prefix(self, prefix: str) -> int: ...

    def __len__(self) -> int: ...


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"向量维度不一致：{len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class InMemoryVectorStore:
    """单机内存实现，用于测试与临时会话；进程退出即清空。"""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectors: list[list[float]] = []
        self._ids: set[str] = set()

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks 与 vectors 数量不一致")
        for chunk, vector in zip(chunks, vectors):
            if chunk.id in self._ids:
                continue
            self._ids.add(chunk.id)
            self._chunks.append(chunk)
            self._vectors.append(list(vector))

    def search(self, vector: Sequence[float], top_k: int = 5) -> list[ScoredChunk]:
        scored = [
            ScoredChunk(chunk=chunk, score=_cosine(vector, vec), detail={"vector": _cosine(vector, vec)})
            for chunk, vec in zip(self._chunks, self._vectors)
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[: max(1, top_k)]

    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def vectors(self) -> list[list[float]]:
        return list(self._vectors)

    def clear(self) -> None:
        self._chunks.clear()
        self._vectors.clear()
        self._ids.clear()

    def delete_by_source_prefix(self, prefix: str) -> int:
        keep_chunks: list[Chunk] = []
        keep_vectors: list[list[float]] = []
        removed = 0
        for chunk, vector in zip(self._chunks, self._vectors):
            if chunk.source.startswith(prefix):
                removed += 1
                self._ids.discard(chunk.id)
            else:
                keep_chunks.append(chunk)
                keep_vectors.append(vector)
        self._chunks, self._vectors = keep_chunks, keep_vectors
        return removed

    def close(self) -> None:  # 与 SqliteVectorStore 对齐，便于统一调用
        return None

    def __len__(self) -> int:
        return len(self._chunks)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    seq    INTEGER PRIMARY KEY AUTOINCREMENT,
    id     TEXT    NOT NULL UNIQUE,
    text   TEXT    NOT NULL,
    source TEXT    NOT NULL DEFAULT '',
    meta   TEXT    NOT NULL DEFAULT '{}',
    dim    INTEGER NOT NULL,
    vector BLOB    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_id ON chunks(id);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class SqliteVectorStore:
    """SQLite 持久化向量库。重启、换机器（拷贝 db 文件）都能带走全部卡片。

    - 向量以 float32 BLOB 存储，加载时拼矩阵并预归一化，检索用矩阵乘
    - 自增 seq 保证 chunks() 顺序稳定——检索器的 BM25 文档靠这个顺序对齐
    - 换 EMBEDDER（如 hash 512 维 → bge-m3 1024 维）时会显式抛错而不是静默乱算
    """

    def __init__(self, path: str | Path, *, dim: int = 0) -> None:
        self.path = Path(path)
        if self.path.parent and not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False：FastAPI 的同步路由跑在线程池里，连接会跨线程使用。
        # 写操作（入库/清空）全部串行化，检索走内存矩阵不碰连接，因此不需要连接池。
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None
        self._dim = dim
        self._reload()

    # ---------- 内部 ----------
    def _reload(self) -> None:
        rows = self._conn.execute(
            "SELECT id, text, source, meta, dim, vector FROM chunks ORDER BY seq"
        ).fetchall()
        self._chunks = []
        vectors: list[np.ndarray] = []
        for cid, text, source, meta, dim, blob in rows:
            try:
                meta_dict = json.loads(meta or "{}")
            except json.JSONDecodeError:
                meta_dict = {}
            self._chunks.append(
                Chunk(id=cid, text=text, source=source or "", meta=meta_dict or {})
            )
            vectors.append(np.frombuffer(blob, dtype=np.float32))
            self._dim = int(dim)

        if not vectors:
            self._matrix = None
            return

        dims = {int(v.shape[0]) for v in vectors}
        if len(dims) > 1:
            raise ValueError(
                f"库内向量维度不一致 {sorted(dims)}，通常是换过 EMBEDDER 未重建索引。"
                "请删除 data/index 下的 db 文件后重新入库。"
            )
        matrix = np.vstack(vectors).astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._matrix = matrix / norms

    # ---------- 写入 ----------
    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks 与 vectors 数量不一致")

        # 先算好在锁外完成 np 转换，锁内只做数据库写入
        prepared: list[tuple] = []
        for chunk, vector in zip(chunks, vectors):
            arr = np.asarray(vector, dtype=np.float32).ravel()
            prepared.append(
                (
                    chunk.id,
                    chunk.text,
                    chunk.source or "",
                    json.dumps(chunk.meta or {}, ensure_ascii=False),
                    int(arr.shape[0]),
                    arr.tobytes(),
                )
            )

        with self._lock:
            existing = {row[0] for row in self._conn.execute("SELECT id FROM chunks")}
            rows: list[tuple] = []
            dim = self._dim
            for row in prepared:
                cid, _text, _source, _meta, dim_of_row, _blob = row
                if cid in existing:
                    continue
                if dim and int(dim_of_row) != dim:
                    raise ValueError(
                        f"向量维度 {dim_of_row} 与库内已有 {dim} 不一致："
                        "换过 EMBEDDER 后必须清空索引重建（删除 data/index 下的 db 文件，或调用 clear()）"
                    )
                rows.append(row)
                existing.add(cid)
                dim = int(dim_of_row)

            if not rows:
                return
            self._conn.executemany(
                "INSERT INTO chunks(id, text, source, meta, dim, vector) VALUES(?,?,?,?,?,?)",
                rows,
            )
            self._conn.commit()
            self._reload()

    # ---------- 检索 ----------
    def search(self, vector: Sequence[float], top_k: int = 5) -> list[ScoredChunk]:
        if self._matrix is None:
            return []
        query = np.asarray(vector, dtype=np.float32).ravel()
        if int(query.shape[0]) != int(self._matrix.shape[1]):
            raise ValueError(
                f"查询向量维度 {query.shape[0]} 与库内 {self._matrix.shape[1]} 不一致，"
                "EMBEDDER 与索引不匹配"
            )
        norm = float(np.linalg.norm(query)) or 1.0
        sims = self._matrix @ (query / norm)
        order = np.argsort(-sims)[: max(1, top_k)]
        return [
            ScoredChunk(
                chunk=self._chunks[int(i)],
                score=float(sims[int(i)]),
                detail={"vector": float(sims[int(i)])},
            )
            for i in order
        ]

    # ---------- 读取 / 维护 ----------
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def vectors(self) -> list[list[float]]:
        return [] if self._matrix is None else self._matrix.tolist()

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM chunks")
            self._conn.commit()
            self._reload()

    # ---------- 元信息（记录索引由哪个向量模型建立） ----------
    def get_meta(self, key: str, default: str = "") -> str:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else default

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, str(value)),
            )
            self._conn.commit()

    def delete_by_source_prefix(self, prefix: str) -> int:
        """按来源前缀删除卡片。改资料、清理冒烟数据时用。"""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM chunks WHERE source LIKE ?", (f"{prefix}%",)
            )
            self._conn.commit()
            deleted = cur.rowcount or 0
            if deleted:
                self._reload()
            return deleted

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass

    def __len__(self) -> int:
        return len(self._chunks)


def build_store(name: str = "sqlite", path: str | Path | None = None) -> VectorStore:
    """按配置构造向量库。sqlite 必须给路径，避免悄悄落到未知位置。"""
    key = (name or "").strip().lower()
    if key in {"memory", "mem", "ram"}:
        return InMemoryVectorStore()
    if key in {"sqlite", "sqlite3", "db"}:
        if path is None:
            raise ValueError("VECTOR_STORE=sqlite 时必须提供 db 路径")
        return SqliteVectorStore(path)
    raise ValueError(f"未知的 VECTOR_STORE: {name}，可选 memory / sqlite")
