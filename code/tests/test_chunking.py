"""切块测试：卡片必须可独立检索，不能丢上下文。"""
from feiyi.rag.chunking import Chunk, chunk_text, split_sentences

LONG_TEXT = (
    "皮影戏又称灯影戏，起源于西汉，盛行于唐宋。"
    "皮影的制作要经过选皮、制皮、画稿、镂刻、敷彩、缀结六道工序。"
    "选皮多用牛皮或驴皮，制皮时要把皮子浸泡刮薄晾干，使皮面平整透光。"
    "镂刻是最关键的一步，艺人用数十把刻刀在皮面上刻出人物轮廓与纹样。"
    "敷彩使用矿物颜料，色彩透明，久不褪色。"
    "缀结时以线连缀各关节，使其能够活动。"
)


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_short_text_yields_single_chunk():
    chunks = chunk_text("秦腔是陕西地方戏曲。", size=300, source="秦腔简介")
    assert len(chunks) == 1
    assert chunks[0].source == "秦腔简介"
    assert "秦腔" in chunks[0].text


def test_long_text_is_split_into_multiple_chunks():
    chunks = chunk_text(LONG_TEXT, size=80, overlap=0, source="皮影戏")
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 90  # size + 单句溢出余量


def test_chunks_have_unique_ids():
    chunks = chunk_text(LONG_TEXT, size=60, overlap=10, source="皮影戏")
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


def test_adjacent_chunks_overlap():
    chunks = chunk_text(LONG_TEXT, size=80, overlap=20, source="皮影戏")
    assert len(chunks) >= 2
    tail = chunks[0].text[-10:]
    assert tail in chunks[1].text, "相邻块应保留重叠内容，避免句子被切断后语义丢失"


def test_very_long_sentence_is_hard_split():
    sentence = "工序" * 200  # 400 字，无标点
    chunks = chunk_text(sentence, size=100, overlap=0, source="压力测试")
    assert len(chunks) == 4
    assert all(len(c.text) <= 100 for c in chunks)


def test_split_sentences_keeps_punctuation():
    sentences = split_sentences("第一步选皮。第二步制皮！第三步画稿？")
    assert sentences == ["第一步选皮。", "第二步制皮！", "第三步画稿？"]


def test_chunk_dataclass_defaults():
    chunk = Chunk(id="x", text="内容")
    assert chunk.source == ""
    assert chunk.meta == {}
