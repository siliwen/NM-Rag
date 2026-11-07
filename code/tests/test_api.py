"""接口测试：四个核心接口必须端到端可用（离线桩，不联网、不占显存）。"""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from feiyi.api import create_app
from feiyi.config import Settings

SAMPLE = (
    "皮影戏又称灯影戏，起源于西汉，盛行于唐宋。"
    "选皮多用牛皮或驴皮，制皮时要把皮子浸泡刮薄晾干，使皮面平整透光。"
    "秦腔是陕西地方戏曲，唱腔高亢激越。"
)


@pytest.fixture()
def client() -> TestClient:
    settings = Settings()
    settings.embedder = "hash"
    settings.llm_api_key = ""
    return TestClient(create_app(settings))


def parse_ndjson(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_health_reports_empty_index(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["chunks"] == 0
    assert body["llm_configured"] is False


def test_index_then_search_hits(client):
    resp = client.post("/knowledge/index", json={"text": SAMPLE, "source": "皮影戏申报书", "size": 80})
    assert resp.status_code == 200
    assert resp.json()["indexed"] > 0

    search = client.post("/knowledge/search", json={"query": "牛皮怎么处理", "top_k": 2})
    assert search.status_code == 200
    hits = search.json()["hits"]
    assert hits
    assert "牛皮" in hits[0]["text"]
    assert hits[0]["source"] == "皮影戏申报书"


def test_index_rejects_empty_text(client):
    resp = client.post("/knowledge/index", json={"text": "   ", "source": "空"})
    assert resp.status_code == 422


def test_say_streams_events(client):
    client.post("/knowledge/index", json={"text": SAMPLE, "source": "皮影戏申报书", "size": 80})
    session_id = client.post("/session").json()["session_id"]

    resp = client.post(f"/session/{session_id}/say", json={"text": "牛皮怎么处理"})
    assert resp.status_code == 200
    events = parse_ndjson(resp.text)
    kinds = [e["type"] for e in events]
    assert "sources" in kinds
    assert "token" in kinds
    assert kinds[-1] == "state"
    assert events[-1]["state"] == "idle"


def test_say_on_unknown_session_returns_404(client):
    resp = client.post("/session/not-exist/say", json={"text": "你好"})
    assert resp.status_code == 404


def test_interrupt_endpoint(client):
    session_id = client.post("/session").json()["session_id"]
    resp = client.post(f"/session/{session_id}/interrupt")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert resp.json()["interrupts"] == 1


def test_duplicate_index_does_not_grow(client):
    first = client.post("/knowledge/index", json={"text": SAMPLE, "source": "同一份", "size": 80})
    second = client.post("/knowledge/index", json={"text": SAMPLE, "source": "同一份", "size": 80})
    assert first.json()["indexed"] > 0
    assert second.json()["indexed"] == 0


def test_seed_loads_demo_corpus(client):
    resp = client.post("/knowledge/seed", json={"reset": True})
    assert resp.status_code == 200
    body = resp.json()
    assert body["indexed"] > 0
    assert body["total"] == body["indexed"]


def test_seed_with_reset_does_not_accumulate(client):
    client.post("/knowledge/seed", json={"reset": True})
    first = client.get("/health").json()["chunks"]
    second = client.post("/knowledge/seed", json={"reset": True}).json()
    assert second["total"] == first, "reset=true 时不应累加卡片"


def test_eval_endpoint_returns_metrics(client):
    resp = client.post("/eval/run")
    assert resp.status_code == 200
    body = resp.json()
    for key in ("hit1", "hit3", "mrr", "by_level", "oos", "failures"):
        assert key in body
    assert 0.0 <= body["hit1"] <= 1.0
    assert body["by_level"]["L4"]["total"] > 0, "评测集必须有完全越界题"
    assert body["by_level"]["L5"]["total"] > 0, "评测集必须有「相关但无事实」题"


# ---------------------------------------------------------------- PDF 上传
SAMPLE_PDF = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "samples"
    / "pdf"
    / "十四五非物质文化遗产保护规划.pdf"
)
needs_sample_pdf = pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="缺少样例 PDF")


def upload_pdf(client, path: Path, **form):
    form.setdefault("size", 200)
    form.setdefault("max_pages", 2)  # 用例只跑前两页，保持快
    with path.open("rb") as fh:
        return client.post(
            "/knowledge/index-pdf",
            files={"file": (path.name, fh, "application/pdf")},
            data=form,
        )


@needs_sample_pdf
def test_index_pdf_returns_page_stats(client):
    resp = upload_pdf(client, SAMPLE_PDF)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["pages"] == 2
    assert body["indexed"] > 0
    assert body["source"] == SAMPLE_PDF.name
    assert body["ocr_pages"] == [], "文字版 PDF 不该触发 OCR"


@needs_sample_pdf
def test_pdf_chunks_are_searchable_with_page_number(client):
    upload_pdf(client, SAMPLE_PDF)
    hits = client.post("/knowledge/search", json={"query": "非物质文化遗产保护", "top_k": 3}).json()["hits"]
    assert hits
    assert isinstance(hits[0]["page"], int)
    assert hits[0]["page"] >= 1


@needs_sample_pdf
def test_pdf_reupload_replaces_instead_of_accumulating(client):
    first = upload_pdf(client, SAMPLE_PDF).json()
    second = upload_pdf(client, SAMPLE_PDF).json()
    assert second["replaced"] == first["indexed"]
    assert second["total"] == first["total"], "同名来源重复上传不能翻倍"


def test_index_pdf_rejects_non_pdf(client):
    resp = client.post(
        "/knowledge/index-pdf",
        files={"file": ("假的.pdf", b"this is not a pdf", "application/pdf")},
    )
    assert resp.status_code == 422
    assert "解析失败" in str(resp.json()["detail"])


def test_index_pdf_rejects_empty_file(client):
    resp = client.post(
        "/knowledge/index-pdf",
        files={"file": ("空.pdf", b"", "application/pdf")},
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------- 音频入库
def _write_wav(path, seconds=2.0):
    import numpy as np
    import soundfile as sf

    sr = 16000
    sf.write(str(path), (0.2 * np.sin(2 * np.pi * 220 * np.arange(int(sr * seconds)) / sr)).astype("float32"), sr)
    return path


@pytest.fixture()
def audio_client(tmp_path) -> TestClient:
    """注入桩引擎：接口测试绝不能加载 900MB 模型。"""
    from feiyi.rag.audio import Segment, StubAsrEngine

    settings = Settings()
    settings.embedder = "hash"
    settings.llm_api_key = ""
    app = create_app(settings)
    app.state.container.asr = StubAsrEngine(
        segments=[
            Segment(0, 2000, "我叫王守义，今年六十七了。"),
            Segment(9000, 12000, "刮皮这个活儿最见功夫，手上一用劲就刮穿了。"),
        ]
    )
    return TestClient(app)


@pytest.fixture()
def sample_wav(tmp_path):
    return _write_wav(tmp_path / "口述史片段.wav", seconds=13.0)


def upload_audio(client, path, **extra):
    form = {"size": 300, "gap_ms": 1200, "speaker": "false", "max_seconds": 0, "replace": "true"}
    form.update({k: str(v) for k, v in extra.items()})
    with path.open("rb") as fh:
        return client.post(
            "/knowledge/index-audio",
            files={"file": (path.name, fh, "audio/wav")},
            data=form,
        )


def test_index_audio_reports_time_range(audio_client, sample_wav):
    resp = upload_audio(audio_client, sample_wav)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["indexed"] == 2, "停顿 7 秒，应切成两张卡片"
    assert body["segments"] == 2
    assert body["speakers"] == []
    assert body["chunks_preview"][0]["at"] == 0


def test_audio_chunks_searchable_with_start_ms(audio_client, sample_wav):
    upload_audio(audio_client, sample_wav)
    hits = audio_client.post("/knowledge/search", json={"query": "刮皮最见功夫", "top_k": 2}).json()["hits"]
    assert hits
    assert hits[0]["start_ms"] == 9000
    assert hits[0]["end_ms"] == 12000
    assert hits[0]["meta"]["kind"] == "audio"


def test_audio_reupload_replaces_instead_of_accumulating(audio_client, sample_wav):
    first = upload_audio(audio_client, sample_wav).json()
    second = upload_audio(audio_client, sample_wav).json()
    assert second["replaced"] == first["indexed"]
    assert second["total"] == first["total"], "同名录音重复上传不能翻倍"


def test_audio_accepts_corrected_transcript_without_asr(tmp_path):
    """校对稿回灌：完全不跑识别，也不该需要模型。"""
    settings = Settings()
    settings.embedder = "hash"
    settings.llm_api_key = ""
    client = TestClient(create_app(settings))  # asr 保持 None

    transcript = "# 时长：00:20\n[00:02] 我叫王守义，今年六十七了。\n[00:12] 这门手艺不能在我手上断了。\n"
    resp = client.post(
        "/knowledge/index-audio",
        files={"file": ("口述史.txt", b"", "text/plain")},
        data={"source": "王守义口述史", "transcript": transcript},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["indexed"] >= 1
    assert body["engine"] == "human"
    hits = client.post("/knowledge/search", json={"query": "手艺不能断", "top_k": 1}).json()["hits"]
    # 校对稿只有起始时间、没有结束时间，所以两段之间算不出停顿，会合成一张卡片。
    # 这是刻意的取舍：宁可卡片长一点，也不要凭空编一个结束时间。
    assert hits[0]["start_ms"] == 2000
    assert "手艺不能" in hits[0]["text"]


def test_audio_rejects_unsupported_extension(audio_client):
    resp = audio_client.post(
        "/knowledge/index-audio",
        files={"file": ("录音.xyz", b"whatever", "application/octet-stream")},
    )
    assert resp.status_code == 415
    assert "不支持" in resp.json()["detail"]


def test_audio_rejects_empty_file(audio_client):
    resp = audio_client.post(
        "/knowledge/index-audio",
        files={"file": ("空.wav", b"", "audio/wav")},
    )
    assert resp.status_code == 400


def test_audio_rejects_corrupt_audio(audio_client, tmp_path):
    bad = tmp_path / "坏文件.wav"
    bad.write_bytes(b"not really audio")
    resp = upload_audio(audio_client, bad)
    assert resp.status_code == 422


def test_audio_no_speech_returns_422_not_silent_success(tmp_path):
    """识别不出内容必须报错并给提示，否则等于入库 0 张卡片还报成功。"""
    from feiyi.rag.audio import StubAsrEngine

    settings = Settings()
    settings.embedder = "hash"
    settings.llm_api_key = ""
    app = create_app(settings)
    app.state.container.asr = StubAsrEngine(segments=[])  # 一句都不返回

    client = TestClient(app)
    wav = _write_wav(tmp_path / "静音.wav", seconds=1.0)
    resp = upload_audio(client, wav)
    assert resp.status_code == 422
    assert "没有识别出语音内容" in str(resp.json()["detail"])


def test_audio_rejects_transcript_without_sentences(audio_client):
    """给了校对稿但里面一句正文都没有：必须明确报错，不能当成功。"""
    resp = audio_client.post(
        "/knowledge/index-audio",
        files={"file": ("x.wav", b"", "audio/wav")},
        data={"transcript": "   \n# 只有注释\n"},
    )
    assert resp.status_code == 422
    assert "没有解析出" in resp.json()["detail"]
