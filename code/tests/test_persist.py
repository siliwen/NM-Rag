"""持久化测试：向量库落盘、重启后数据与检索能力都要恢复。

为什么单独测 BM25 重建：检索器的 `_doc_tokens` 与 `store.chunks()` 靠 zip 对齐，
持久化后如果不重建，向量还在但关键词打分全为 0，检索质量会在无声无息中退化。
"""
from __future__ import annotations

import pytest

from feiyi.rag.chunking import Chunk
from feiyi.rag.embedder import HashingEmbedder
from feiyi.rag.retriever import HybridRetriever
from feiyi.rag.store import SqliteVectorStore, build_store

DIM = 512


def _chunks() -> list[Chunk]:
    return [
        Chunk(id="a", text="皮影戏起源于西汉，盛行于唐宋。", source="皮影戏起源"),
        Chunk(id="b", text="选皮多用牛皮或驴皮，制皮时要浸泡刮薄晾干。", source="皮影制皮"),
        Chunk(id="c", text="秦腔唱腔高亢激越，用梆子击节。", source="秦腔简介"),
    ]


def _vectors() -> list[list[float]]:
    return HashingEmbedder(dim=DIM).embed([c.text for c in _chunks()])


def test_sqlite_store_survives_reopen(tmp_path):
    db = tmp_path / "knowledge.db"

    first = SqliteVectorStore(db)
    first.add(_chunks(), _vectors())
    assert len(first) == 3
    first.close()

    second = SqliteVectorStore(db)
    assert len(second) == 3
    # 顺序必须与写入一致，否则 BM25 的文档对齐会错位
    assert [c.id for c in second.chunks()] == ["a", "b", "c"]
    assert second.chunks()[1].text == "选皮多用牛皮或驴皮，制皮时要浸泡刮薄晾干。"
    assert second.chunks()[1].source == "皮影制皮"
    second.close()


def test_sqlite_store_is_idempotent_on_same_id(tmp_path):
    store = SqliteVectorStore(tmp_path / "knowledge.db")
    store.add(_chunks(), _vectors())
    store.add(_chunks(), _vectors())
    assert len(store) == 3
    store.close()


def test_sqlite_store_rejects_dimension_mismatch(tmp_path):
    store = SqliteVectorStore(tmp_path / "knowledge.db")
    store.add([Chunk(id="a", text="皮影")], [[0.1, 0.2, 0.3]])
    with pytest.raises(ValueError, match="维度"):
        store.add([Chunk(id="b", text="秦腔")], [[0.1, 0.2]])
    store.close()


def test_sqlite_store_search_ranks_by_cosine(tmp_path):
    store = SqliteVectorStore(tmp_path / "knowledge.db")
    store.add(_chunks(), _vectors())
    query_vec = HashingEmbedder(dim=DIM).embed(["牛皮怎么刮薄"])[0]
    hits = store.search(query_vec, top_k=3)
    assert len(hits) == 3
    assert hits[0].chunk.id == "b"
    assert hits[0].score >= hits[-1].score
    store.close()


def test_sqlite_store_persists_meta(tmp_path):
    db = tmp_path / "knowledge.db"
    store = SqliteVectorStore(db)
    store.add([Chunk(id="m", text="皮影", meta={"page": 12, "项目": "皮影戏"})], [[0.5, 0.5]])
    store.close()

    again = SqliteVectorStore(db)
    assert again.chunks()[0].meta == {"page": 12, "项目": "皮影戏"}
    again.close()


def test_retriever_rebuilds_bm25_after_reopen(tmp_path):
    """重启后不仅向量能查到，关键词分量也必须恢复。"""
    db = tmp_path / "knowledge.db"
    embedder = HashingEmbedder(dim=DIM)

    first = HybridRetriever(embedder=embedder, store=SqliteVectorStore(db))
    first.index(_chunks())
    assert len(first) == 3
    first.store.close()

    second = HybridRetriever(embedder=embedder, store=SqliteVectorStore(db))
    assert len(second) == 3

    hits = second.search("牛皮要怎么处理", top_k=3)
    assert hits[0].chunk.id == "b"
    assert hits[0].detail["keyword"] > 0, "重启后 BM25 索引未重建，关键词打分退化为 0"

    second.store.close()


def test_api_survives_restart(tmp_path):
    """整个应用重启后，已入库的卡片数量不变。"""
    from fastapi.testclient import TestClient

    from feiyi.api import create_app
    from feiyi.config import Settings

    cfg = Settings(embedder="hash", vector_store="sqlite", data_dir=str(tmp_path))
    client1 = TestClient(create_app(cfg))
    body = {"text": "皮影戏起源于西汉。选皮多用牛皮或驴皮，制皮时要浸泡刮薄晾干。", "source": "皮影", "size": 80}
    assert client1.post("/knowledge/index", json=body).status_code == 200
    count = client1.get("/health").json()["chunks"]
    assert count >= 1

    client2 = TestClient(create_app(cfg))
    assert client2.get("/health").json()["chunks"] == count


def test_build_store_returns_memory_by_default(tmp_path):
    store = build_store("memory")
    store.add([Chunk(id="a", text="皮影")], [[1.0, 0.0]])
    assert len(store) == 1


def test_build_store_sqlite_uses_given_path(tmp_path):
    target = tmp_path / "index" / "knowledge.db"
    store = build_store("sqlite", target)
    store.add([Chunk(id="a", text="皮影")], [[1.0, 0.0]])
    store.close()
    assert target.exists()


def test_build_store_rejects_unknown_backend():
    with pytest.raises(ValueError, match="未知"):
        build_store("milvus")


def test_sqlite_store_delete_by_source_prefix(tmp_path):
    db = tmp_path / "knowledge.db"
    store = SqliteVectorStore(db)
    store.add(_chunks(), _vectors())
    assert store.delete_by_source_prefix("皮影") == 2
    assert len(store) == 1
    assert store.chunks()[0].id == "c"
    store.close()

    # 删除要落盘
    again = SqliteVectorStore(db)
    assert len(again) == 1
    again.close()


def test_purge_api_removes_and_rebuilds_index(tmp_path):
    """删卡片后 BM25 文档必须同步收缩，否则后续检索会对齐到错误的文本。"""
    from fastapi.testclient import TestClient

    from feiyi.api import create_app
    from feiyi.config import Settings

    cfg = Settings(embedder="hash", vector_store="sqlite", data_dir=str(tmp_path))
    client = TestClient(create_app(cfg))
    for source, text in [
        ("冒烟测试-A", "选皮多用牛皮，制皮时要浸泡刮薄晾干。"),
        ("冒烟测试-B", "秦腔唱腔高亢激越，用梆子击节。"),
        ("正式资料", "皮影戏起源于西汉，盛行于唐宋。"),
    ]:
        client.post("/knowledge/index", json={"text": text, "source": source, "size": 300})
    assert client.get("/health").json()["chunks"] >= 3

    res = client.post("/knowledge/purge", json={"source_prefix": "冒烟测试"}).json()
    assert res["deleted"] >= 2
    assert res["total"] == 1

    hits = client.post("/knowledge/search", json={"query": "秦腔", "top_k": 3}).json()["hits"]
    assert all("冒烟测试" not in h["source"] for h in hits)



class _FakeEmbedder:
    """维度与 hash 相同（512）但向量完全不同，用来模拟「换了模型却看不出维度差异」。"""

    def __init__(self, model_id: str = "fake/model-v2") -> None:
        self.model_id = model_id
        self.dim = DIM

    def embed(self, texts):
        return [[1.0] + [0.0] * (DIM - 1) for _ in texts]


def test_store_records_embedder_fingerprint(tmp_path):
    store = SqliteVectorStore(tmp_path / "k.db")
    store.add(_chunks(), _vectors())
    store.set_meta("embedder", "hash|512")
    store.close()

    reopened = SqliteVectorStore(tmp_path / "k.db")
    assert reopened.get_meta("embedder") == "hash|512", "指纹必须跨重启保留"
    reopened.close()


def test_container_reembeds_when_embedder_changes(tmp_path):
    """换模型后必须用原文重算向量。

    这是最危险的一类问题：hash 与 bge-small-zh 都是 512 维，只比维度发现不了，
    程序照常运行、不报任何错，但检索结果全是垃圾。
    """
    from feiyi.api import Container
    from feiyi.config import Settings

    cfg = Settings(embedder="hash", vector_store="sqlite", data_dir=str(tmp_path))
    first = Container(cfg, embedder=HashingEmbedder(dim=DIM))
    first.retriever.index(_chunks())
    before = first.store.vectors()[0]
    first.close()

    second = Container(cfg, embedder=_FakeEmbedder())
    assert len(second.retriever) == 3, "卡片不能因换模型而丢失"
    after = second.store.vectors()[0]
    assert after != before, "向量必须被重算"
    assert second.store.get_meta("embedder") == "fake/model-v2|" + str(DIM)
    second.close()


def test_container_reembeds_when_fingerprint_missing(tmp_path):
    """老版本建的库没有指纹，向量来源不可信，必须重算。

    hash 与 bge-small-zh 同为 512 维，只比维度发现不了，会静默用错向量。
    """
    from feiyi.api import Container
    from feiyi.config import Settings

    cfg = Settings(embedder="hash", vector_store="sqlite", data_dir=str(tmp_path))
    first = Container(cfg, embedder=HashingEmbedder(dim=DIM))
    first.retriever.index(_chunks())
    first.store._conn.execute("DELETE FROM meta")  # 模拟旧版库：有向量、无指纹
    first.store._conn.commit()
    before = first.store.vectors()[0]
    first.close()

    second = Container(cfg, embedder=_FakeEmbedder())
    assert len(second.retriever) == 3
    assert second.store.vectors()[0] != before
    second.close()
