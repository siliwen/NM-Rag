"""音频转写：把传承人口述史录音变成带时间戳的可检索卡片。

与 PDF 那条链路是同一个套路，但坑完全不一样：

1. **解码不能靠系统 ffmpeg** —— 交付现场是高校的 Windows 机器，装没装 ffmpeg 没人能保证。
   这里用 `av`（PyAV，BSD-3，wheel 内自带 ffmpeg 解码器），`pip install` 完就能读
   mp3/m4a/wav/flac/aac，不要求用户去配 PATH。
2. **必须重采样到 16k 单声道** —— 模型只认 16k。直接拿 44.1k 的字节喂进去不会报错，
   出来的是一段听不懂的噪音，属于「不报错的错」。立体声也不能只取左声道，
   要下混平均，否则只录到一半的声音。
3. **VAD 把整段判成静音时，转写结果是空字符串** —— 直接入库就是零张卡片，
   程序一切正常、库里什么都没有。必须显式告警（NO-006）。
4. **方言是真实短板** —— `paraformer-zh` 是普通话模型，陕西关中话识别率会明显下降。
   所以这条链路**不假设 ASR 结果能直接用**：转写稿可导出成带时间戳的纯文本，
   人工改完再回灌入库。`[00:01:23]` 前缀就是回读时恢复时间戳的依据。

时间戳全部用**毫秒整数**（`start_ms` / `end_ms`）：FunASR 的 `sentence_info`
本身就是毫秒，混用秒会在长音频上放大成分钟级偏差。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np

from .chunking import Chunk, make_id

TARGET_SR = 16000
DEFAULT_CHUNK_SECONDS = 0  # 0 = 不裁剪，整段转写
DEFAULT_TAIL_GAP_MS = 1200  # 相邻语音段间隔超过此值，认为换了话题，切开
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".opus", ".wma", ".amr"}

# 转写稿里的一行时间戳：[00:01:23] 或 [01:23]
_TS_LINE = re.compile(r"^\s*\[(?:(\d{1,3}):)?(\d{1,2}):(\d{2})\]\s*(.*)$")
# 转写稿头部记的总时长，回读时用来补最后一段的结束时间
_DURATION_LINE = re.compile(r"^#\s*时长[:：]\s*(?:(\d{1,3}):)?(\d{1,2}):(\d{2})")
# 校对稿里的说话人前缀（多说话人时写入的）
_SPEAKER_PREFIX = re.compile(r"^(说话人\d+)\s*[:：]\s*")


class AudioError(RuntimeError):
    """音频处理失败。必须在接口层原样上报，禁止吞掉（NO-006）。"""


class AsrUnavailableError(AudioError):
    """需要 ASR 但依赖没装。绝不静默返回空文本，否则等于入库了零张卡片还没人知道。"""


# ---------------------------------------------------------------- 数据模型
@dataclass
class Segment:
    """一段连续语音。时间戳为毫秒，相对整段音频开头。"""

    start_ms: int
    end_ms: int
    text: str
    speaker: str | None = None

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)


@dataclass
class Transcript:
    """一份录音的转写结果。"""

    source: str
    duration_ms: int
    segments: list[Segment] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    engine: str = ""
    sample_rate: int = TARGET_SR
    # 波形层面的静音区间（毫秒）。切块靠它判断话题切换，
    # 因为识别结果的时间差在跨静音处会是 0（标点模型把两句并成了一句）。
    pauses: list[tuple[int, int]] = field(default_factory=list)

    @property
    def chars(self) -> int:
        return sum(len(s.text) for s in self.segments)

    @property
    def speech_ms(self) -> int:
        return sum(s.duration_ms for s in self.segments)

    @property
    def speakers(self) -> list[str]:
        seen: list[str] = []
        for seg in self.segments:
            if seg.speaker and seg.speaker not in seen:
                seen.append(seg.speaker)
        return seen

    def text(self) -> str:
        return "".join(s.text for s in self.segments if s.text.strip())

    def stats(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "duration_ms": self.duration_ms,
            "duration": fmt_time(self.duration_ms),
            "segments": len(self.segments),
            "chars": self.chars,
            "speech_ms": self.speech_ms,
            "speech_ratio": round(self.speech_ms / self.duration_ms, 3) if self.duration_ms else 0.0,
            "speakers": self.speakers,
            "engine": self.engine,
            "warnings": list(self.warnings),
        }


def fmt_time(ms: int | float) -> str:
    """毫秒 → `03:21` 或 `1:03:21`（给人看的时间点）。

    刻意用**向下取整**而不是四舍五入：取整会把 3.4 秒报成 `00:03`，
    而四舍五入会报成 `00:03`→`00:04`，用户点过去时已经越过这句话的开头了。
    宁可比实际早半秒，也不要晚。
    """
    total = max(0, int(float(ms) // 1000))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------- 解码
def decode_audio(path: str | Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """解码任意常见音频为 float32 单声道，并重采样到 target_sr。

    返回 (samples, sample_rate)，samples 为 1-D float32，取值范围 [-1, 1]。
    """
    path = Path(path)
    if not path.exists():
        raise AudioError(f"文件不存在：{path}")
    if path.stat().st_size == 0:
        raise AudioError(f"文件是空的：{path.name}")

    try:
        import av
    except ImportError as exc:  # pragma: no cover - 环境问题
        raise AudioError("缺少 av（PyAV）：pip install av") from exc

    try:
        container = av.open(str(path))
    except Exception as exc:
        # PyAV 对损坏文件抛的异常类型不稳定，统一包一层并保留原因
        raise AudioError(f"无法打开音频（格式不支持或文件损坏）：{path.name} → {exc}") from exc

    try:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise AudioError(f"文件里没有音频轨：{path.name}")

        # 重采样器一次搞定「声道下混 + 采样率转换 + 样本格式」，避免手写平均
        resampler = av.AudioResampler(format="flt", layout="mono", rate=target_sr)
        blocks: list[np.ndarray] = []
        try:
            for frame in container.decode(stream):
                for out in resampler.resample(frame):
                    blocks.append(out.to_ndarray())
            # 冲掉重采样器内部残留的尾巴，否则长音频末尾会丢几十毫秒
            for out in resampler.resample(None):
                blocks.append(out.to_ndarray())
        except AudioError:
            raise
        except Exception as exc:
            raise AudioError(f"音频解码失败：{path.name} → {exc}") from exc
    finally:
        container.close()

    if not blocks:
        raise AudioError(f"解码后没有得到任何音频数据：{path.name}")

    data = np.concatenate([b.reshape(-1) for b in blocks]).astype(np.float32, copy=False)
    if data.size == 0:
        raise AudioError(f"解码后音频长度为 0：{path.name}")
    return data, target_sr


def audio_duration_ms(samples: np.ndarray, sample_rate: int) -> int:
    return int(round(len(samples) / float(sample_rate) * 1000))


def detect_pauses(
    samples: np.ndarray,
    sample_rate: int = TARGET_SR,
    *,
    frame_ms: int = 20,
    min_silence_ms: int = 300,
    floor_multiplier: float = 2.5,
) -> list[tuple[int, int]]:
    """从波形里找出静音区间（毫秒），用来判断「这里是不是换了话题」。

    为什么不直接用识别出来的句子时间差：标点模型会把跨静音的两句合并成一句，
    合并后两段时间戳首尾相接、时间差是 **0**，看起来"毫无停顿"。
    实测就踩到了——段落之间明明有 1.5 秒静音，时间差却是 0。

    阈值用**相对噪声底**（10 分位能量 × 倍数），不用固定值：
    现场录音有底噪，固定阈值会把整段都判成静音；相对阈值在安静和嘈杂环境都能用。
    """
    if samples.size == 0:
        return []
    hop = max(1, int(sample_rate * frame_ms / 1000))
    usable = (samples.size // hop) * hop
    if usable < hop * 2:
        return []
    frames = samples[:usable].reshape(-1, hop).astype(np.float32, copy=False)
    energy = np.sqrt(np.mean(frames**2, axis=1))

    speech = float(np.percentile(energy, 90))
    if speech <= 1e-8:
        # 整段几乎没有声音：谈不上"停顿"，交给上层按「没识别到内容」处理
        return []

    floor = float(np.percentile(energy, 5))
    # 阈值封顶在语音能量的 35%：不加这个上限，一段没有静音的录音会把
    # 十分位能量（≈语音本身）当成噪声底，乘 2.5 后超过所有帧，整段被判成静音。
    threshold = min(floor * floor_multiplier, speech * 0.35)
    quiet = energy <= threshold

    min_frames = max(1, int(min_silence_ms / frame_ms))
    candidates: list[tuple[int, int]] = []
    start = None
    for i, is_quiet in enumerate(quiet):
        if is_quiet and start is None:
            start = i
        elif not is_quiet and start is not None:
            if i - start >= min_frames:
                candidates.append((start, i))
            start = None
    if start is not None and len(quiet) - start >= min_frames:
        candidates.append((start, len(quiet)))

    pauses: list[tuple[int, int]] = []
    for s, e in candidates:
        # 再验一道：真静音的能量必须远低于语音，不能只是"相对安静"
        if float(np.median(energy[s:e])) <= speech * 0.2:
            pauses.append((s * frame_ms, e * frame_ms))
    return pauses


def _silence_ratio(samples: np.ndarray, sample_rate: int, threshold: float = 1e-4) -> float:
    """粗估静音占比（按 100ms 一帧算峰值）。用来给「整段没声音」兜底告警。"""
    hop = max(1, sample_rate // 10)
    if samples.size < hop:
        return 1.0 if float(np.max(np.abs(samples))) <= threshold else 0.0
    frames = samples[: (samples.size // hop) * hop].reshape(-1, hop)
    quiet = np.max(np.abs(frames), axis=1) <= threshold
    return float(quiet.mean())


# ---------------------------------------------------------------- 引擎接口
class AsrEngine(Protocol):
    """识别引擎。抽象出来是为了测试不必真的加载 840MB 模型。"""

    name: str

    def transcribe(self, samples: np.ndarray, sample_rate: int = TARGET_SR) -> list[Segment]: ...


class StubAsrEngine:
    """离线桩：按固定文本平均切段，用于串联链路与单元测试。

    刻意保留「按音频长度等比分配时间戳」的行为——测试切块逻辑时，
    时间戳必须随音频长度变化，否则测不出停顿断开的分支。
    """

    name = "stub"

    def __init__(self, text: str = "", segments: Sequence[Segment] | None = None) -> None:
        self.text = text
        # None 与空列表必须区分：None = 没配置，用默认文本兜底；
        # [] = 明确表示「一句都没识别出来」，这是要触发告警的场景。
        self._segments = None if segments is None else list(segments)

    def transcribe(self, samples: np.ndarray, sample_rate: int = TARGET_SR) -> list[Segment]:
        if self._segments is not None:
            return [Segment(s.start_ms, s.end_ms, s.text, s.speaker) for s in self._segments]
        total = audio_duration_ms(samples, sample_rate)
        text = self.text or "（桩引擎：未配置识别文本）"
        return [Segment(0, total, text)]


class FunAsrEngine:
    """FunASR 实现：paraformer-zh（中文）+ fsmn-vad（切句）+ ct-punc（加标点）。

    懒加载：进程启动不碰模型，第一次真转写时才下载/载入。
    """

    name = "funasr"

    def __init__(
        self,
        model: str = "paraformer-zh",
        vad_model: str = "fsmn-vad",
        punc_model: str = "ct-punc",
        spk_model: str = "",
        device: str = "cpu",
        batch_size_s: int = 300,
    ) -> None:
        self.model = model
        self.vad_model = vad_model
        self.punc_model = punc_model
        self.spk_model = spk_model
        self.device = device
        self.batch_size_s = batch_size_s
        self._impl: Any = None

    def _load(self) -> Any:
        if self._impl is None:
            try:
                from funasr import AutoModel
            except ImportError as exc:
                raise AsrUnavailableError(
                    "需要语音识别但依赖未安装：pip install funasr（首次运行还要下载模型约 900MB）"
                ) from exc
            kwargs: dict[str, Any] = {
                "model": self.model,
                "device": self.device,
                # 关掉联网版本检查：交付环境可能没有外网，检查失败会拖慢甚至报错
                "disable_update": True,
            }
            if self.vad_model:
                kwargs["vad_model"] = self.vad_model
            if self.punc_model:
                kwargs["punc_model"] = self.punc_model
            if self.spk_model:
                # 说话人分离：口述史常是「采访人 + 传承人」两人对答
                kwargs["spk_model"] = self.spk_model
            try:
                self._impl = AutoModel(**kwargs)
            except Exception as exc:
                raise AsrUnavailableError(
                    f"FunASR 模型加载失败（{self.model}）：{exc}"
                ) from exc
        return self._impl

    def load(self) -> "FunAsrEngine":
        """显式预加载模型，让调用方能把「加载」和「转写」的耗时分开统计。

        不分开统计的话，转写一段 45 秒音频会显示「耗时 62 秒、RTF 1.39」——
        其中 58 秒是在加载模型，这个数字会让使用者误判硬件够不够用。
        """
        self._load()
        return self

    def transcribe(self, samples: np.ndarray, sample_rate: int = TARGET_SR) -> list[Segment]:
        if sample_rate != TARGET_SR:
            raise AudioError(
                f"ASR 引擎只接受 {TARGET_SR}Hz 输入，收到 {sample_rate}Hz；请先用 decode_audio 重采样"
            )
        model = self._load()
        duration = audio_duration_ms(samples, sample_rate)
        try:
            # sentence_timestamp=True 是必须的：不传的话 FunASR 只回一段整段文本
            # 加词级 timestamp，**没有 sentence_info**，等于丢掉了全部时间点。
            # 这个坑很隐蔽——结果看起来"成功"（有 text、不报错），
            # 只是每张卡片的时间戳都退化成 0，用户点过去永远跳到开头。
            result = model.generate(
                input=samples,
                batch_size_s=self.batch_size_s,
                sentence_timestamp=True,
            )
        except Exception as exc:
            raise AudioError(f"语音识别失败：{exc}") from exc

        if not result:
            return []
        item = result[0]
        raw_segments = _segments_from_result(item)
        if not raw_segments:
            text = str(item.get("text") or "").strip()
            if not text:
                return []
            # 没开 VAD 或没给 sentence_info 时，只有整段文本，时间戳只能用全长兜底
            return [Segment(0, duration, text)]

        return normalize_segments(raw_segments, duration)


def _segments_from_result(item: dict[str, Any]) -> list[Segment]:
    """从 FunASR 返回里取「句级 + 时间戳」结果。

    两种结构都要认：
    - 开了 vad/punc：`sentence_info` = [{start, end, text, spk?}, ...]（毫秒）
    - 只给了 `timestamp`：[[start_ms, end_ms, token], ...] 的词级时间戳
    """
    out: list[Segment] = []
    for seg in item.get("sentence_info") or []:
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        spk = seg.get("spk")
        out.append(
            Segment(
                start_ms=int(seg.get("start") or 0),
                end_ms=int(seg.get("end") or 0),
                text=text,
                speaker=f"说话人{int(spk) + 1}" if spk is not None else None,
            )
        )
    return out


def normalize_segments(segments: Sequence[Segment], duration_ms: int) -> list[Segment]:
    """收拾模型给的脏数据：倒序、负时长、越界、空文本。

    模型偶尔会给出 end < start 或时间戳超出音频长度（尤其长音频尾部）。
    放任不管的后果是卡片带一个荒唐的时间点，用户点过去发现音频里没有这句话。
    """
    cleaned: list[Segment] = []
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        start = max(0, int(seg.start_ms))
        end = int(seg.end_ms)
        if end <= start:
            end = start
        if duration_ms > 0:
            start = min(start, duration_ms)
            end = min(end, duration_ms)
        cleaned.append(Segment(start, end, text, seg.speaker))

    cleaned.sort(key=lambda s: (s.start_ms, s.end_ms))
    # 后一段起点不得早于前一段终点，避免时间轴上出现倒退
    prev_end = 0
    for seg in cleaned:
        if seg.start_ms < prev_end:
            seg.start_ms = prev_end
            if seg.end_ms < seg.start_ms:
                seg.end_ms = seg.start_ms
        prev_end = seg.end_ms
    return cleaned


# ---------------------------------------------------------------- 主流程
def transcribe_audio(
    path: str | Path,
    *,
    source: str = "",
    engine: AsrEngine | None = None,
    max_seconds: int = DEFAULT_CHUNK_SECONDS,
    on_progress: Callable[[str], None] | None = None,
) -> Transcript:
    """整段转写一份录音。

    参数
    ----
    engine : 不传则用 FunAsrEngine（首次会下载模型）
    max_seconds : >0 时只转写前 N 秒（长篇口述史先试听一段用）
    """
    if engine is None:
        engine = FunAsrEngine()

    name = source or Path(path).name
    samples, sr = decode_audio(path)
    duration = audio_duration_ms(samples, sr)

    if max_seconds and max_seconds > 0 and duration > max_seconds * 1000:
        samples = samples[: max_seconds * sr]
        duration = audio_duration_ms(samples, sr)

    quiet = _silence_ratio(samples, sr)
    tr = Transcript(
        source=name,
        duration_ms=duration,
        engine=getattr(engine, "name", "unknown"),
        sample_rate=sr,
        pauses=detect_pauses(samples, sr),
    )

    if on_progress is not None:
        on_progress(f"解码完成：{fmt_time(duration)}，静音占比 {quiet:.0%}")

    segments = engine.transcribe(samples, sr)
    resolved_duration = duration
    tr.segments = normalize_segments(segments, resolved_duration)

    if not tr.segments:
        tr.warnings.append(
            "未识别到任何语音内容：可能是整段静音、纯噪声，或语言/模型不匹配。"
            "已跳过入库，不会产生空白卡片。"
        )
    elif quiet >= 0.9:
        tr.warnings.append(f"音频里 {quiet:.0%} 是静音，识别结果可能不完整")
    if max_seconds and max_seconds > 0 and duration >= max_seconds * 1000:
        tr.warnings.append(f"只转写了前 {fmt_time(max_seconds * 1000)}（max_seconds={max_seconds}）")

    return tr


def _has_pause_between(
    pauses: Sequence[tuple[int, int]],
    a_end: int,
    b_start: int,
    min_ms: int,
    tolerance: int = 300,
) -> bool:
    """两段之间有没有一段足够长的静音。

    判据是「静音**起点**落在两段的交界处」，不是「静音中点落在两段之间」：
    标点模型把跨静音的两句合并后，两段时间戳首尾相接（`b_start == a_end`），
    「中点落在区间内」这种写法会直接判否，把最需要切开的场景漏掉。
    留 300ms 容差，是因为能量检测的静音边界和模型时间戳本来就不会严丝合缝。
    """
    if not pauses:
        return False
    for pause_start, pause_end in pauses:
        if pause_end - pause_start < min_ms:
            continue
        if (a_end - tolerance) <= pause_start <= (b_start + tolerance):
            return True
    return False


def transcript_chunks(
    tr: Transcript,
    *,
    size: int = 300,
    gap_ms: int = DEFAULT_TAIL_GAP_MS,
    source: str = "",
    keep_speaker: bool = True,
    use_pauses: bool = True,
) -> list[Chunk]:
    """把转写结果合成卡片，每张带时间段。

    两个切分条件，缺一不可：
    - **字数**：跟文本链路一致，单张卡片 300 字左右，避免检索命中后塞给大模型一大段。
    - **停顿**：口述史里长停顿基本等于换话题，强行接上一张卡片会把「学艺经历」和
      「纹样寓意」粘成一条，检索时互相污染。

    停顿的判定有两个来源，任何一个成立就切开：
    1. 相邻识别段之间的时间差 > `gap_ms`（简单直接）；
    2. **波形里的静音区间**落在两段之间且长度 > `gap_ms`。
    第 2 条是必需的：标点模型会把跨静音的两句合并，合并后时间差是 0，
    只靠第 1 条会漏掉这种"假连续"——实测就漏了段落边界。

    卡片 id 用全局递增序号生成（同 PDF 那条链路的理由）：
    按段各自编号会让不同位置算出同一个 id，入库时被 store 静默去重。
    """
    src = source or tr.source
    segments = [s for s in tr.segments if s.text.strip()]
    if not segments:
        return []

    multi_speaker = keep_speaker and len(tr.speakers) > 1
    pauses = tr.pauses if use_pauses else []

    groups: list[list[Segment]] = []
    current: list[Segment] = []
    current_len = 0
    for seg in segments:
        if current:
            prev = current[-1]
            over_budget = current_len + len(seg.text) > size and current_len >= size // 2
            long_pause = (seg.start_ms - prev.end_ms > gap_ms) or _has_pause_between(
                pauses, prev.end_ms, seg.start_ms, gap_ms
            )
            if over_budget or long_pause or (multi_speaker and seg.speaker != prev.speaker):
                groups.append(current)
                current, current_len = [], 0
        current.append(seg)
        current_len += len(seg.text)
    if current:
        groups.append(current)

    out: list[Chunk] = []
    for group in groups:
        if multi_speaker:
            body = "".join(
                f"{s.speaker}：{s.text}" if s.speaker else s.text for s in group
            )
        else:
            body = "".join(s.text for s in group)
        text = body.strip()
        if not text:
            continue
        start_ms = group[0].start_ms
        end_ms = group[-1].end_ms
        meta: dict[str, Any] = {
            "kind": "audio",
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_ms": max(0, end_ms - start_ms),
            "asr": tr.engine,
        }
        speakers = [s.speaker for s in group if s.speaker]
        if speakers:
            meta["speaker"] = speakers[0]
        out.append(Chunk(id=make_id(src, len(out), text), text=text, source=src, meta=meta))
    return out


def transcribe_to_chunks(
    path: str | Path,
    *,
    source: str = "",
    engine: AsrEngine | None = None,
    size: int = 300,
    gap_ms: int = DEFAULT_TAIL_GAP_MS,
    max_seconds: int = DEFAULT_CHUNK_SECONDS,
) -> tuple[Transcript, list[Chunk]]:
    """一步到位：转写 + 切块。CLI 与服务端都走这里，避免两套行为漂移。"""
    tr = transcribe_audio(path, source=source, engine=engine, max_seconds=max_seconds)
    return tr, transcript_chunks(tr, size=size, gap_ms=gap_ms, source=source or tr.source)


# ---------------------------------------------------------------- 转写稿（人工校对闭环）
def format_transcript(tr: Transcript, *, header: bool = True) -> str:
    """导出成「带时间戳的纯文本」，给人改错字用。

    刻意不用 JSON：传承人讲的是方言，校对的人多半不是程序员，
    给一份能直接在任何编辑器里改的文本，比给结构化数据现实得多。
    """
    lines: list[str] = []
    if header:
        lines.append(f"# 来源：{tr.source}")
        lines.append(f"# 时长：{fmt_time(tr.duration_ms)}　引擎：{tr.engine}　段数：{len(tr.segments)}")
        lines.append("# 直接用编辑器改错字即可；行首的 [时间戳] 不要删，回灌时靠它恢复时间点。")
        if tr.warnings:
            for w in tr.warnings:
                lines.append(f"# 警告：{w}")
        lines.append("")
    for seg in tr.segments:
        prefix = f"{seg.speaker}：" if seg.speaker else ""
        lines.append(f"[{fmt_time(seg.start_ms)}] {prefix}{seg.text}")
    return "\n".join(lines) + ("\n" if lines else "")


def parse_transcript(text: str, *, source: str = "", engine: str = "human") -> Transcript:
    """回读校对稿。时间戳按行首 `[MM:SS]` / `[HH:MM:SS]` 恢复。

    没有时间戳的行会被并到上一段（编辑时不小心敲了回车不该丢内容）；
    出现在第一行之前则从 0 开始。

    最后一段的结束时间从文件头的「# 时长：」补回来。
    不补的话，只要人工改过稿，尾段的时长就会退化成 0，
    卡片上的时间段变成「12:03 - 12:03」，等于告诉用户这段只有一个瞬间。
    """
    segments: list[Segment] = []
    last_ms = 0
    total_ms = 0

    for raw in (text or "").splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            m_dur = _DURATION_LINE.match(stripped)
            if m_dur:
                h = int(m_dur.group(1) or 0)
                total_ms = ((h * 60 + int(m_dur.group(2))) * 60 + int(m_dur.group(3))) * 1000
            continue

        m = _TS_LINE.match(line)
        if m:
            h = int(m.group(1) or 0)
            mi = int(m.group(2))
            sec = int(m.group(3))
            last_ms = ((h * 60 + mi) * 60 + sec) * 1000
            body = m.group(4).strip()
        else:
            body = stripped

        speaker = None
        m_spk = _SPEAKER_PREFIX.match(body)
        if m_spk:
            speaker = m_spk.group(1)
            body = body[m_spk.end() :].strip()

        if not body:
            continue
        if segments and not m:
            segments[-1].text += body
        else:
            segments.append(Segment(last_ms, last_ms, body, speaker))

    # 单段时长用「下一段起点」推；最后一段用文件头总时长兜底
    for i, seg in enumerate(segments):
        if i + 1 < len(segments):
            seg.end_ms = max(seg.start_ms, segments[i + 1].start_ms)
        else:
            seg.end_ms = max(seg.start_ms, total_ms)

    duration = total_ms or (segments[-1].end_ms if segments else 0)
    return Transcript(source=source, duration_ms=duration, segments=segments, engine=engine)


def load_transcript(path: str | Path) -> Transcript:
    """读取校对稿：`.jsonl` 按段读，其他按带时间戳的纯文本读。"""
    p = Path(path)
    if not p.exists():
        raise AudioError(f"转写稿不存在：{p}")
    raw = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".jsonl":
        segments: list[Segment] = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AudioError(f"转写稿第 {len(segments) + 1} 行不是合法 JSON：{exc}") from exc
            segments.append(
                Segment(
                    start_ms=int(obj.get("start_ms") or 0),
                    end_ms=int(obj.get("end_ms") or 0),
                    text=str(obj.get("text") or ""),
                    speaker=obj.get("speaker"),
                )
            )
        duration = max((s.end_ms for s in segments), default=0)
        return Transcript(source=p.stem, duration_ms=duration, segments=segments, engine="human")
    return parse_transcript(raw, source=p.stem)


def save_transcript(tr: Transcript, path: str | Path) -> Path:
    """导出校对稿（默认纯文本；`.jsonl` 后缀导出结构化版本）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".jsonl":
        lines = [
            json.dumps(
                {
                    "start_ms": s.start_ms,
                    "end_ms": s.end_ms,
                    "text": s.text,
                    **({"speaker": s.speaker} if s.speaker else {}),
                },
                ensure_ascii=False,
            )
            for s in tr.segments
        ]
        p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    else:
        p.write_text(format_transcript(tr), encoding="utf-8")
    return p
