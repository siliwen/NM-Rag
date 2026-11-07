"""生成「模拟口述史」测试音频，**同时产出标准答案**。

为什么要自己造样本，而不是只下载现成的：
网上能找到的非遗音频要么没有对应文本，要么只有几秒钟。
没有标准答案，转写质量就只能靠肉眼"看着差不多"——测不出字准率，
也测不出时间戳偏了几秒。这个脚本合成时就知道每句话落在第几毫秒，
于是 CER 和时间戳误差都能算出来。

合成用 Windows 自带的 SAPI（Huihui 中文音色），**只用于造测试素材**。
交付版的 TTS 仍然是本地 CosyVoice2（见 AGENT.md NO-009），两者无关。

刻意保留口述史的真实特征：
- 段落之间留 1.5 秒长停顿 —— 检验切块会不会按话题断开
- 句间留 0.3 秒短停顿 —— 不应被切开
- 可用 --noise-db 叠白噪声，模拟现场录音
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from feiyi.rag.audio import TARGET_SR, decode_audio  # noqa: E402

# 口述史正文：第一人称，讲皮影戏的学艺与制作，带一点关中口语
PARAGRAPHS: list[list[str]] = [
    [
        "我叫王守义，今年六十七了，家住陕西华县，就是现在叫渭南市华州区。",
        "我们家唱皮影戏，到我这一辈是第四代。",
        "我十一岁那年跟着我爹学，那时候不叫学艺，叫帮忙，先是在后台递签子、捡影人。",
    ],
    [
        "皮影戏的影人，是用牛皮刻出来的。",
        "选皮要选三岁口的黄牛皮，太老的皮子发脆，太嫩的又立不住。",
        "皮子泡软了以后要刮，刮到能透光，薄得跟纸一样，还不能破。",
        "刮皮这个活儿最见功夫，手上一用劲就刮穿了，破了就废了。",
        "刮好的皮子绷在木框上阴干，干透了才能画稿子。",
    ],
    [
        "画稿子讲究一个线条连贯，一笔下去不能断。",
        "刻的时候手里握着十几把刀，刀口宽窄都不一样，刻头发丝用最细的那把。",
        "刻一个武生，光是脸谱上的花纹就要一天工夫。",
        "刻完了要染色，颜料是矿物的，用桐油调开，染一遍晾一遍，前后要染七八遍。",
        "最后一道是上油，上了油影人才透亮，灯光一打，红是红绿是绿。",
    ],
    [
        "唱戏的调子是碗碗腔，乐器简单，就一把月琴、一副梆子、一个碗碗。",
        "碗碗就是个小铜碗，拿筷子敲，声音清脆，所以叫碗碗腔。",
        "一个人能唱好几个角色，男的女的都得会，嗓子要能翻上去也能压下来。",
        "现在年轻人愿意学的不多了，我带了三个徒弟，最长的学了六年。",
        "我就想着，这门手艺不能在我手上断了。",
    ],
]

INTRA_GAP_MS = 300
INTER_GAP_MS = 1500


def synth_sentence(text: str, voice: str, rate: int) -> np.ndarray:
    """用 SAPI 把一句话合成成 16k 单声道 float32。"""
    import win32com.client  # pywin32，仅 Windows

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "line.wav"
        speaker = win32com.client.Dispatch("SAPI.SpVoice")
        for item in speaker.GetVoices():
            if voice and voice.lower() in item.GetDescription().lower():
                speaker.Voice = item
                break
        speaker.Rate = rate
        stream = win32com.client.Dispatch("SAPI.SpFileStream")
        # 3 = SSFMCreateForWrite
        stream.Open(str(wav_path), 3, False)
        speaker.AudioOutputStream = stream
        speaker.Speak(text)
        stream.Close()
        if not wav_path.exists():
            raise RuntimeError(f"语音合成没有产生文件：{text[:20]}")
        samples, _sr = decode_audio(wav_path, target_sr=TARGET_SR)
        return samples


def build_sample(
    out_wav: Path,
    out_truth: Path,
    *,
    voice: str = "Huihui",
    rate: int = 0,
    noise_db: float = 0.0,
    repeat: int = 1,
) -> dict:
    pieces: list[np.ndarray] = []
    truth: list[dict] = []
    cursor_ms = 0
    index = 0

    rounds = max(1, repeat)
    for r_i in range(rounds):
        for p_i, para in enumerate(PARAGRAPHS):
            for s_i, sentence in enumerate(para):
                is_last = r_i == rounds - 1 and p_i == len(PARAGRAPHS) - 1 and s_i == len(para) - 1
                audio = synth_sentence(sentence, voice, rate)
                dur_ms = int(round(len(audio) / TARGET_SR * 1000))
                truth.append(
                    {
                        "index": index,
                        "text": sentence,
                        "start_ms": cursor_ms,
                        "end_ms": cursor_ms + dur_ms,
                        "paragraph": p_i + 1,
                    }
                )
                index += 1
                pieces.append(audio)
                cursor_ms += dur_ms

                gap = 0 if is_last else (INTER_GAP_MS if s_i == len(para) - 1 else INTRA_GAP_MS)
                if gap > 0:
                    pieces.append(np.zeros(int(TARGET_SR * gap / 1000), dtype=np.float32))
                    cursor_ms += gap

    audio = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)
    if noise_db > 0:
        rms = float(np.sqrt(np.mean(audio**2))) or 1e-3
        noise = np.random.default_rng(20261105).normal(0, rms / (10 ** (noise_db / 20)), audio.size)
        audio = (audio + noise).astype(np.float32)

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_wav), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(TARGET_SR)
        fh.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())

    payload = {
        "source": out_wav.name,
        "sample_rate": TARGET_SR,
        "duration_ms": int(round(len(audio) / TARGET_SR * 1000)),
        "chars": sum(len(t["text"]) for t in truth),
        "sentences": truth,
        "note": "SAPI 合成的模拟口述史，仅作测试素材；文本即标准答案。",
    }
    out_truth.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # 自检：时长必须和标准答案对得上，否则说明拼接逻辑错了
    expect = truth[-1]["end_ms"] if truth else 0
    actual = payload["duration_ms"]
    if abs(actual - expect) > 1200:  # 尾部最后一句话后面还有停顿
        raise SystemExit(f"自检失败：音频时长 {actual}ms 与标准答案 {expect}ms 相差过大")

    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="生成模拟口述史测试音频（含标准答案）")
    ap.add_argument("--out", default="oral_history_sample.wav")
    ap.add_argument("--out-truth", default="")
    ap.add_argument("--voice", default="Huihui", help="SAPI 音色关键词，默认 Huihui（中文女声）")
    ap.add_argument("--rate", type=int, default=0, help="语速 -10~10，负数更慢（更接近老人语速）")
    ap.add_argument("--noise-db", type=float, default=0.0, help="叠白噪声的 SNR(dB)，0 表示不加")
    ap.add_argument("--repeat", type=int, default=1, help="正文重复遍数，>1 用来造长音频")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    truth = Path(args.out_truth).resolve() if args.out_truth else out.with_suffix(".truth.json")
    payload = build_sample(out, truth, voice=args.voice, rate=args.rate, noise_db=args.noise_db, repeat=args.repeat)

    print(f"[完成] {out.name}")
    print(f"  时长 {payload['duration_ms'] / 1000:.1f}s ｜ {len(payload['sentences'])} 句 ｜ {payload['chars']} 字")
    print(f"  标准答案：{truth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
