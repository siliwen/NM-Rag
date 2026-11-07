"""检索测试：非遗专名必须能被准确召回，无关内容不能混进 top。"""
from feiyi.rag.chunking import Chunk
from feiyi.rag.embedder import HashingEmbedder, build_embedder
from feiyi.rag.retriever import HybridRetriever, tokenize
from feiyi.rag.store import InMemoryVectorStore

DOCS = [
    Chunk(id="c1", text="皮影戏起源于西汉，盛行于唐宋，是民间广为流传的傀儡艺术。", source="皮影戏起源"),
    Chunk(id="c2", text="选皮多用牛皮或驴皮，制皮时要把皮子浸泡刮薄晾干，使皮面平整透光。", source="皮影制皮"),
    Chunk(id="c3", text="镂刻是最关键的一步，艺人用数十把刻刀在皮面上刻出人物轮廓与纹样。", source="皮影镂刻"),
    Chunk(id="c4", text="秦腔是陕西地方戏曲，唱腔高亢激越，常用梆子击节伴奏。", source="秦腔简介"),
]


def build_retriever() -> HybridRetriever:
    return HybridRetriever(embedder=HashingEmbedder(), keyword_weight=0.6)


def test_tokenize_produces_chinese_bigrams():
    tokens = tokenize("皮影戏 ABC123")
    assert "皮影" in tokens
    assert "影戏" in tokens
    assert "abc123" in tokens


def test_index_returns_added_count():
    retriever = build_retriever()
    assert retriever.index(DOCS) == 4
    assert len(retriever) == 4


def test_index_is_idempotent():
    retriever = build_retriever()
    retriever.index(DOCS)
    assert retriever.index(DOCS) == 0
    assert len(retriever) == 4


def test_query_hits_relevant_chunk_first():
    retriever = build_retriever()
    retriever.index(DOCS)
    hits = retriever.search("牛皮要怎么处理才能做皮影", top_k=2)
    assert hits, "检索不应为空"
    assert hits[0].chunk.id == "c2", f"应命中断皮工序，实际命中 {hits[0].chunk.source}"


def test_unrelated_query_scores_low():
    retriever = build_retriever()
    retriever.index(DOCS)
    hits = retriever.search("今天西安天气怎么样", top_k=1)
    assert hits[0].score < 0.2


def test_scores_are_normalized():
    retriever = build_retriever()
    retriever.index(DOCS)
    for hit in retriever.search("镂刻用刻刀", top_k=4):
        assert 0.0 <= hit.score <= 1.0
        assert "vector" in hit.detail and "keyword" in hit.detail


def test_empty_query_returns_empty():
    retriever = build_retriever()
    retriever.index(DOCS)
    assert retriever.search("", top_k=3) == []


def test_search_on_empty_index_returns_empty():
    assert build_retriever().search("皮影", top_k=3) == []


def test_custom_store_is_used():
    store = InMemoryVectorStore()
    retriever = HybridRetriever(embedder=HashingEmbedder(), store=store)
    retriever.index(DOCS)
    assert len(store) == 4


def test_build_embedder_rejects_unknown_name():
    import pytest

    with pytest.raises(ValueError):
        build_embedder("不存在的模型")


def test_out_of_scope_query_cannot_score_high():
    """越界问题不能因为「有字样重叠」就拿到高分。

    旧实现把关键词分按本次查询的最大值归一化，等于任何有字样重叠的查询，
    其最优结果都会被抬到 1.0，于是「食堂几点开门」也能拿 0.6 以上，
    分数阈值就失去了拒答能力。改用饱和函数后必须显著降低。
    """
    retriever = build_retriever()
    retriever.index(DOCS)
    hits = retriever.search("你们食堂几点开门", top_k=1)
    assert hits[0].score < 0.35, f"越界问题得分过高：{hits[0].score}"


def test_in_scope_hit_keeps_strong_keyword_signal():
    """降低越界分不能把正常命中一起压低。"""
    retriever = build_retriever()
    retriever.index(DOCS)
    hits = retriever.search("制皮要把皮子刮薄晾干", top_k=1)
    assert hits[0].chunk.id == "c2"
    assert hits[0].detail["keyword"] > 0.5, "正常命中的关键词分不应被压平"
