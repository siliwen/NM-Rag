"""评测核心测试：指标算错比没有指标更糟，所以指标本身要盯住。"""
import json

from feiyi.eval import build_corpus, evaluate, read_jsonl
from feiyi.rag.embedder import HashingEmbedder


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return path


CORPUS_ROWS = [
    {"source": "皮影制皮", "text": "选皮多用牛皮或驴皮，制皮时要把皮子浸泡刮薄晾干。"},
    {"source": "皮影镂刻", "text": "镂刻是最关键的一步，艺人用数十把刻刀在皮面上刻出纹样。"},
    {"source": "秦腔唱腔", "text": "秦腔唱腔高亢激越，常用梆子击节伴奏。"},
]

QA_ROWS = [
    {"q": "牛皮要怎么处理", "expect": "皮影制皮", "level": "L1"},
    {"q": "镂刻用什么工具", "expect": "皮影镂刻", "level": "L2"},
    {"q": "食堂几点开门", "expect": None, "level": "L4"},
]


def test_metrics_count_perfect_ranking(tmp_path):
    corpus = build_corpus(write_jsonl(tmp_path / "c.jsonl", CORPUS_ROWS))
    qa = read_jsonl(write_jsonl(tmp_path / "q.jsonl", QA_ROWS))
    result = evaluate(HashingEmbedder(), corpus, qa)
    assert result["hit1"] == 1.0
    assert result["mrr"] == 1.0
    assert result["by_level"]["L1"]["hit"] == 1
    assert result["by_level"]["L4"]["total"] == 1


def test_wrong_answer_is_counted_as_failure(tmp_path):
    corpus = build_corpus(write_jsonl(tmp_path / "c.jsonl", CORPUS_ROWS))
    qa = read_jsonl(
        write_jsonl(
            tmp_path / "q.jsonl",
            [{"q": "牛皮要怎么处理", "expect": "完全不存在的条目", "level": "L2"}],
        )
    )
    result = evaluate(HashingEmbedder(), corpus, qa)
    assert result["hit1"] == 0.0
    assert result["failures"][0]["rank"] == 0


def test_out_of_scope_does_not_count_as_hit(tmp_path):
    corpus = build_corpus(write_jsonl(tmp_path / "c.jsonl", CORPUS_ROWS))
    qa = read_jsonl(
        write_jsonl(tmp_path / "q.jsonl", [{"q": "今天天气", "expect": None, "level": "L4"}])
    )
    result = evaluate(HashingEmbedder(), corpus, qa)
    assert result["hit1"] == 0.0, "L4 越界题不能计入命中率分母"
    assert result["oos"]["L4"]["n"] == 1


def test_hit3_counts_when_correct_card_is_second(tmp_path):
    corpus = build_corpus(write_jsonl(tmp_path / "c.jsonl", CORPUS_ROWS))
    qa = read_jsonl(
        write_jsonl(
            tmp_path / "q.jsonl",
            [{"q": "皮影 镂刻 秦腔 牛皮", "expect": "皮影镂刻", "level": "L2"}],
        )
    )
    result = evaluate(HashingEmbedder(), corpus, qa)
    assert result["hit3"] == 1.0, "正确答案在前 3 位就算 top-3 命中"
    assert result["hit1"] <= 1.0


def test_read_jsonl_skips_comments_and_blanks(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text('# 注释\n\n{"a": 1}\n', encoding="utf-8")
    assert read_jsonl(path) == [{"a": 1}]
