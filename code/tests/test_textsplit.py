"""切句逻辑单测：整段切分与流式切分两套入口。"""
from feiyi.textsplit import pop_complete_sentences, split_sentences


def test_split_keeps_punctuation_and_splits_on_sentence_end():
    assert split_sentences("第一句话。第二句话！第三句话？") == [
        "第一句话。",
        "第二句话！",
        "第三句话？",
    ]


def test_split_treats_newline_as_boundary():
    parts = split_sentences("第一段内容在第一行。\n\n第二段内容在第二行。")
    assert len(parts) == 2
    assert all("\n" not in p for p in parts)


def test_split_merges_tiny_pieces():
    parts = split_sentences("好。这句话稍微长一点点。", min_chars=6)
    assert len(parts) == 1


def test_split_hard_wraps_long_sentence():
    long = "，".join(["这是一个很长的分句" for _ in range(8)]) + "。"
    parts = split_sentences(long, max_chars=20)
    assert len(parts) > 1
    assert all(len(p) <= 20 for p in parts)
    assert "".join(parts) == long


def test_split_returns_empty_for_blank_text():
    assert split_sentences("   \n  ") == []


def test_pop_returns_finished_sentences_and_keeps_remainder():
    done, rest = pop_complete_sentences("第一句说完了。第二句也完了。第三句还没")
    assert done == ["第一句说完了。", "第二句也完了。"]
    assert rest == "第三句还没"


def test_pop_keeps_partial_sentence_untouched():
    done, rest = pop_complete_sentences("这句还没说完")
    assert done == []
    assert rest == "这句还没说完"


def test_pop_holds_back_tiny_sentence_when_more_content_follows():
    done, rest = pop_complete_sentences("好。后面还有一句更长的话。", min_chars=6)
    assert done == ["好。后面还有一句更长的话。"]
    assert rest == ""


def test_pop_emits_tiny_sentence_on_flush():
    # 流已经结束：短句也必须吐出来，否则最后一句话永远合成不了
    done, rest = pop_complete_sentences("好。", flush=True)
    assert done == ["好。"]
    assert rest == ""


def test_pop_holds_tiny_sentence_without_flush():
    done, rest = pop_complete_sentences("好。")
    assert done == []
    assert rest == "好。"


def test_pop_flush_emits_trailing_partial_sentence():
    done, rest = pop_complete_sentences("这句话没有句号", flush=True)
    assert done == ["这句话没有句号"]
    assert rest == ""


def test_pop_cuts_at_comma_when_sentence_longer_than_window():
    """第一段要短：整句 30 多字时先在逗号处切，别等整句合成完才出声。"""
    text = "皮影戏就是用灯光把皮做的影人照在幕布上讲故事，也叫灯影戏。"
    done, rest = pop_complete_sentences(text, min_chars=6, max_chars=26)
    assert done and done[0].endswith("，"), done
    assert len(done[0]) <= 26
    assert "".join(done) + rest == text


def test_pop_keeps_sentence_whole_when_within_window():
    text = "这一句只有二十来个字，不到窗口上限。"
    done, _ = pop_complete_sentences(text, min_chars=4, max_chars=40)
    assert done == [text], "整句没超窗口就不该被拆开"


def test_pop_allow_break_cuts_at_comma_without_waiting_for_period():
    """首段专用：句号还没来，逗号就先开口，别让数字人干等着。"""
    buf = "皮影戏就是用灯光把皮影人偶照在幕布上讲故事，也叫"
    done, rest = pop_complete_sentences(buf, min_chars=6, max_chars=26, allow_break=True)
    assert done == ["皮影戏就是用灯光把皮影人偶照在幕布上讲故事，"]
    assert rest == "也叫"


def test_pop_without_allow_break_still_waits_for_sentence_end():
    buf = "皮影戏就是用灯光把皮影人偶照在幕布上讲故事，也叫"
    done, rest = pop_complete_sentences(buf, min_chars=6, max_chars=26)
    assert done == []
    assert rest == buf


def test_pop_allow_break_respects_min_chars():
    # 逗号出现得太早（不足 min_chars）时不该切，否则合成出来全是碎片
    done, _ = pop_complete_sentences("好，", min_chars=6, max_chars=26, allow_break=True)
    assert done == []


def test_pop_hard_cuts_when_no_punctuation_for_too_long():
    text = "这是一段一直没有句号的很长很长的话" * 3
    done, rest = pop_complete_sentences(text, max_chars=20)
    assert done, "长时间没有句末标点必须硬切，否则数字人一直不出声"
    assert "".join(done) + rest == text.replace(" ", "")


def test_pop_handles_multiple_sentences_in_one_burst():
    done, _ = pop_complete_sentences("一。二二二二。三三三三。", min_chars=2)
    assert done == ["一。", "二二二二。", "三三三三。"]


def test_pop_allow_break_hard_cuts_when_comma_is_beyond_window():
    """首段专用：逗号在窗口之外（或压根没有）时，必须按窗口硬切。

    实测 Kokoro 每字约 50ms：23 字首段要合成 1.6 秒，12 字只要 0.85 秒。
    首段多等一个逗号，数字人就晚开口近 1 秒。
    """
    buf = "皮影戏就是用灯光把皮做的影人照在幕布上讲故事，也叫灯影戏。"
    done, rest = pop_complete_sentences(buf, min_chars=4, max_chars=12, allow_break=True)

    assert done, "首段必须立刻吐出来，不能继续等逗号"
    assert len(done[0]) <= 12, f"首段超过窗口：{len(done[0])} 字 / {done[0]}"
    assert "".join(done) + rest == buf, "切分不能丢字"
