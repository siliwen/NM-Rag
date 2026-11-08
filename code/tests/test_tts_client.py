"""本地 TTS 客户端：绕开系统代理、参数透传、失败显式抛错。"""
import asyncio

import httpx
import pytest

from feiyi.clients.tts import HttpTTS


def test_http_tts_bypasses_proxy_by_default():
    """本机实测：走系统代理会让每次合成多花约 0.6 秒，首音延迟直接翻倍。"""
    assert HttpTTS(base_url="http://127.0.0.1:9880").trust_env is False


def test_http_tts_requires_base_url():
    with pytest.raises(RuntimeError):
        HttpTTS(base_url="")


def test_synthesize_sends_text_and_voice_without_proxy(monkeypatch):
    captured = {}

    class FakeResp:
        status_code = 200
        content = b"WAVDATA"

    class FakeClient:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, json):
            captured["url"] = url
            captured["json"] = json
            return FakeResp()

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    audio = asyncio.run(HttpTTS(base_url="http://127.0.0.1:9880/", voice_id="zf_001").synthesize("你好"))

    assert audio == b"WAVDATA"
    assert captured["url"] == "http://127.0.0.1:9880/tts"
    assert captured["json"] == {"text": "你好", "voice_id": "zf_001"}
    assert captured["kwargs"]["trust_env"] is False


def test_synthesize_surfaces_service_error(monkeypatch):
    class BadResp:
        status_code = 500
        text = "本地 TTS 合成失败：模型没加载"

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, json):
            return BadResp()

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    with pytest.raises(RuntimeError, match="500"):
        asyncio.run(HttpTTS(base_url="http://127.0.0.1:9880").synthesize("你好"))


def test_http_tts_reuses_connection_pool_across_calls(monkeypatch):
    """一段回答要合成 4~6 次；每次都新建客户端就多付 4~6 次 TCP 握手。

    实测：同样 23 字，直连后端 1.34s，经编排层 1.63s —— 差的 0.29s 就是握手。
    """
    created = []

    class FakeResp:
        status_code = 200
        content = b"WAV"

    class FakeClient:
        is_closed = False

        def __init__(self, **kwargs):
            created.append(kwargs)

        async def post(self, url, json):
            return FakeResp()

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    client = HttpTTS(base_url="http://127.0.0.1:9880", voice_id="zf_001")

    async def twice():
        await client.synthesize("第一段。")
        await client.synthesize("第二段。")

    asyncio.run(twice())

    assert len(created) == 1, f"应复用同一个连接池，实际建了 {len(created)} 个"
    assert created[0]["trust_env"] is False, "复用连接也不能把代理放回来"


def test_warmup_builds_connection_and_touches_backend(monkeypatch):
    """启动时预热：把 TCP 握手和模型加载挪到启动阶段，别算进用户的第一次提问。"""
    calls = []

    class FakeResp:
        status_code = 200
        content = b"WAV"

    class FakeClient:
        is_closed = False

        def __init__(self, **kwargs):
            pass

        async def post(self, url, json):
            calls.append(json["text"])
            return FakeResp()

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    tts = HttpTTS(base_url="http://127.0.0.1:9880", voice_id="zf_001")
    asyncio.run(tts.warmup())

    assert len(calls) == 1, "预热应真发一次合成，顺带把后端模型加载进内存"
    assert len(calls[0]) <= 4, "预热文本要极短，别浪费时间合成一整句"


def test_warmup_failure_does_not_break_startup(monkeypatch):
    """后端没起时预热必须静默失败：服务能启动，只是首段慢一点。"""

    class FakeClient:
        is_closed = False

        def __init__(self, **kwargs):
            pass

        async def post(self, url, json):
            raise RuntimeError("连接被拒绝")

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    asyncio.run(HttpTTS(base_url="http://127.0.0.1:9880").warmup())  # 不抛异常
