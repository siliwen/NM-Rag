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


# ---------- 标题行必须开新块（简历类文档的项目名/时间/角色不能被切走）----------
def test_heading_with_period_starts_new_chunk():
    """项目标题行（「XX项目 项目经理 2020.05-2020.12」）若留在上一块末尾，
    检索命中的描述卡就没有项目名，模型只能自己编一个。"""
    text = (
        "上一段讲的是完全无关的交付细节，占掉一些字数用来制造换块压力。\n"
        "成都旺小宝VR虚拟看房平台 项目经理 2020.05-2020.12\n"
        "房地产客户对线上看房有需求，需要可交互的全景场景。"
    )
    chunks = chunk_text(text, size=40, overlap=10, source="简历")
    head = [c for c in chunks if "成都旺小宝" in c.text]

    assert head, "标题行不能凭空丢失"
    assert head[0].text.startswith("成都旺小宝"), f"标题应开新块，实际：{head[0].text}"


def test_heading_chunk_does_not_inherit_previous_overlap():
    """标题开新块时不能带上一块的尾巴，否则块首是无关内容，检索会串味。"""
    text = (
        "这是上一段的收尾文字，讲的是别的事情。\n"
        "宁德时代质量域四位一体项目 项目经理 2024.07-2025.03\n"
        "从零组建并带领本地开发团队，构建数字化管控平台。"
    )
    chunks = chunk_text(text, size=80, overlap=20, source="简历")
    head = [c for c in chunks if "宁德时代" in c.text][0]

    assert head.text.startswith("宁德时代"), f"标题块混入了上一块内容：{head.text}"


def test_period_sentence_is_not_treated_as_heading():
    """正文里的单一年份或带句号的句子不能被误判成标题，否则切块过碎。"""
    text = (
        "2023年负责公司内部PMS系统迭代交付工作，完成约40个迭代版本。"
        "2022年支持教育系统疫情防控平台建设，覆盖3万余名师生。"
    )
    chunks = chunk_text(text, size=200, overlap=0, source="资料")

    assert len(chunks) == 1, f"正文不该被标题规则切碎：{[c.text for c in chunks]}"


def test_heading_detected_when_pdf_lost_newlines():
    """PDF 提取丢换行时，标题后面往往没有标点，与描述黏成一段。

    真实简历：「…讲完了。中电2021镇安县基层治理平台 项目经理 2021.09-2022.01通，事件上报链路长…」
    标题前后的标点都不完整，只在「句号之后、描述之前」有位置可切。
    切不开 → 描述卡没有项目名 → 模型只能编一个。
    """
    text = (
        "上一段的交付细节讲完了，这里再补充几句无关内容把前面的块撑满。"
        "中电2021镇安县基层治理平台 项目经理 2021.09-2022.01"
        "通，事件上报链路长，指挥中心无法实时掌握现场状况，制约了响应速度。"
    )
    chunks = chunk_text(text, size=200, overlap=0, source="简历")
    idx = [i for i, c in enumerate(chunks) if "镇安县" in c.text]

    assert idx, "标题不能丢失"
    head = chunks[idx[0]]
    assert head.text.startswith("中电2021"), f"标题应开新块，实际：{head.text}"


def test_heading_and_its_description_stay_together():
    """标题切出来后必须和它自己的描述在同一块；只剩标题的卡片等于没用。"""
    text = (
        "上一段的交付细节讲完了，这里再补充几句无关内容把前面的块撑满。"
        "中电2021镇安县基层治理平台 项目经理 2021.09-2022.01"
        "通，事件上报链路长，指挥中心无法实时掌握现场状况。"
    )
    chunks = chunk_text(text, size=200, overlap=0, source="简历")
    head = [c for c in chunks if "镇安县" in c.text][0]

    assert "事件上报" in head.text, f"标题与描述被分开了：{head.text}"
