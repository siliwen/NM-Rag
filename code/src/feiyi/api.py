"""HTTP 接口层（FastAPI）。

接口清单：
    GET  /health                      健康检查 + 已入库卡片数
    POST /knowledge/index             文本入库（解析 → 切块 → 向量化）
    POST /knowledge/index-pdf         上传 PDF 入库（文本层优先，扫描页自动 OCR）
    POST /knowledge/index-audio       上传录音入库（口述史转写，带时间戳；可传人工校对稿）
    POST /knowledge/search            检索调试（返回分数构成，便于人工校验）
    POST /session                     创建会话
    POST /session/{id}/say            发一句话，返回 NDJSON 事件流
    POST /session/{id}/interrupt      打断
"""
from __future__ import annotations

import json
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from . import __version__
from .clients.llm import build_llm
from .clients.tts import HttpTTS, StubTTS
from .config import Settings, get_settings
from .pipeline import RAGPipeline
from .rag.chunking import chunk_text
from .rag.audio import (
    AUDIO_EXTS,
    AsrUnavailableError,
    AudioError,
    FunAsrEngine,
    parse_transcript,
    transcribe_audio,
    transcript_chunks,
)
from .rag.embedder import build_embedder
from .rag.pdf import OcrUnavailableError, PdfError, extract_pdf, pdf_chunks
from .rag.retriever import HybridRetriever
from .rag.store import build_store
from .session import Session

# 上传上限：非遗申报书通常几 MB，超过 60MB 基本是整本画册，先挡住
MAX_PDF_BYTES = 60 * 1024 * 1024
# 音频放宽到 300MB：一小时 16k 单声道 wav 约 115MB，口述史动辄一两个小时
MAX_AUDIO_BYTES = 300 * 1024 * 1024


class IndexIn(BaseModel):
    text: str = Field(..., min_length=1, description="原始文本")
    source: str = Field("", description="来源标识，如「皮影戏申报书 p12」")
    size: int = Field(300, ge=80, le=1200)
    overlap: int = Field(50, ge=0, le=300)

    @field_validator("text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = (value or "").strip()
        if not value:
            raise ValueError("text 不能为空白")
        return value

    @field_validator("source")
    @classmethod
    def _strip_source(cls, value: str) -> str:
        return (value or "").strip()


class SearchIn(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(3, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = (value or "").strip()
        if not value:
            raise ValueError("query 不能为空白")
        return value


class SayIn(BaseModel):
    text: str = Field(..., min_length=1)

    @field_validator("text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = (value or "").strip()
        if not value:
            raise ValueError("text 不能为空白")
        return value


class PurgeIn(BaseModel):
    source_prefix: str = Field(..., min_length=1, description="要删除的来源前缀，如「冒烟测试」")

    @field_validator("source_prefix")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = (value or "").strip()
        if not value:
            raise ValueError("source_prefix 不能为空白")
        return value


class SeedIn(BaseModel):
    reset: bool = Field(False, description="是否先清空知识库再灌入（演示时用 true）")


class Container:
    """服务级单例容器。生产替换 store / tts / asr 即可，编排层不动。"""

    def __init__(self, settings: Settings, embedder: Any = None) -> None:
        self.settings = settings
        # 空字符串表示让 sentence-transformers 自己挑（有 CUDA 就用 GPU）
        device = settings.embedder_device.strip() or None
        self.embedder = embedder if embedder is not None else build_embedder(
            settings.embedder, device=device
        )
        # 生产默认 SQLite 持久化：重启不丢卡片。测试用 memory，互不污染。
        self.store = build_store(settings.vector_store, settings.index_db_path)
        self.keyword_weight = settings.resolve_keyword_weight(
            has_semantic_embedder=hasattr(self.embedder, "model_id")
        )
        self.retriever = HybridRetriever(
            embedder=self.embedder,
            store=self.store,
            keyword_weight=self.keyword_weight,
        )
        self.llm = build_llm(settings)
        self.pipeline = RAGPipeline(
            retriever=self.retriever,
            llm=self.llm,
            top_k=settings.retrieve_top_k,
            max_context_chars=settings.max_context_chars,
            max_tokens=settings.llm_max_tokens,
        )
        self.tts = (
            HttpTTS(
                base_url=settings.tts_base_url,
                timeout=settings.tts_timeout,
                voice_id=settings.tts_voice_id,
            )
            if settings.tts_base_url
            else StubTTS()
        )
        self.sessions: dict[str, Session] = {}
        # ASR 懒加载：没人传音频就不该把 900MB 模型读进内存。
        # 测试直接赋一个桩引擎即可，不必碰真实模型。
        self.asr: Any = None
        self._ensure_embedder_matches()

    def get_asr(self, speaker: bool = False) -> Any:
        """取识别引擎。首次调用才真正加载模型（约 5 秒 + 首次下载 900MB）。"""
        if self.asr is None:
            self.asr = FunAsrEngine(spk_model="cam++" if speaker else "")
        return self.asr

    @property
    def embedder_fingerprint(self) -> str:
        return f"{getattr(self.embedder, 'model_id', self.settings.embedder)}|{getattr(self.embedder, 'dim', 0)}"

    def _ensure_embedder_matches(self) -> None:
        """索引与当前向量模型不一致时，用库里的原文重算向量。

        为什么必须管这件事：hash 桩和 bge-small-zh 都是 512 维，只比对维度
        根本发现不了换过模型，旧向量会被继续当成有效数据参与计算——
        程序不报错，检索结果却是垃圾。这种「不报错的错」最难排查，必须拦住。
        """
        store = self.store
        set_meta = getattr(store, "set_meta", None)
        if set_meta is None:
            return
        current = self.embedder_fingerprint
        stored = store.get_meta("embedder")
        # stored 为空表示「库里已有向量，但没记录是谁算的」（老版本建的库）。
        # 这种情况同样不可信：来源不明就重算，宁可多花几秒，也不能拿错向量凑合。
        if len(store) and stored != current:
            chunks = store.chunks()
            origin = stored or "未知来源（旧版索引）"
            print(
                f"[索引迁移] 索引由 {origin} 建立，当前模型为 {current}，"
                f"正在用原文重算 {len(chunks)} 张卡片…"
            )
            store.clear()
            self.retriever.rebuild()
            self.retriever.index(chunks)
            print(f"[索引迁移] 完成，共 {len(self.retriever)} 张卡片")
        set_meta("embedder", current)

    def close(self) -> None:
        self.store.close()


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or get_settings()
    container = Container(cfg)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # 预热放在启动阶段：首段合成不再付 TCP 握手（~0.29s）和后端冷加载（十几秒）
        warm = getattr(container.tts, "warmup", None)
        if callable(warm):
            await warm()
        try:
            yield
        finally:
            closer = getattr(container.tts, "aclose", None)
            if callable(closer):
                await closer()

    app = FastAPI(title="非遗数字人讲解员", version=__version__, lifespan=lifespan)
    app.state.container = container

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "chunks": len(container.retriever),
            "embedder": cfg.embedder,
            "embedder_model": getattr(container.embedder, "model_id", cfg.embedder),
            "embedder_dim": getattr(container.embedder, "dim", 0),
            "store": cfg.vector_store,
            "llm_configured": cfg.llm_configured,
            "tts_configured": bool(cfg.tts_base_url),
            "tts_voice_id": cfg.tts_voice_id or None,
        }

    @app.post("/knowledge/index")
    async def knowledge_index(body: IndexIn) -> dict[str, Any]:
        chunks = chunk_text(body.text, size=body.size, overlap=body.overlap, source=body.source)
        if not chunks:
            raise HTTPException(status_code=400, detail="切块结果为空，请检查文本内容")
        added = container.retriever.index(chunks)
        return {"indexed": added, "total": len(container.retriever), "chunks": [c.id for c in chunks]}

    @app.post("/knowledge/search")
    async def knowledge_search(body: SearchIn) -> dict[str, Any]:
        hits = container.retriever.search(body.query, top_k=body.top_k)
        return {
            "query": body.query,
            "total_indexed": len(container.retriever),
            "hits": [
                {
                    "id": hit.chunk.id,
                    "source": hit.chunk.source,
                    "page": hit.chunk.meta.get("page"),
                    "start_ms": hit.chunk.meta.get("start_ms"),
                    "end_ms": hit.chunk.meta.get("end_ms"),
                    "meta": hit.chunk.meta,
                    "score": round(float(hit.score), 4),
                    "detail": hit.detail,
                    "text": hit.chunk.text,
                }
                for hit in hits
            ],
        }

    @app.post("/knowledge/index-pdf")
    async def knowledge_index_pdf(
        file: UploadFile = File(..., description="PDF 文件"),
        source: str = Form("", description="来源标识，默认取文件名"),
        size: int = Form(300, ge=80, le=1200),
        overlap: int = Form(50, ge=0, le=300),
        ocr: str = Form("auto", description="auto ｜ always ｜ never"),
        replace: bool = Form(True, description="按同名来源覆盖旧卡片，避免重复入库"),
        max_pages: int = Form(0, ge=0, le=5000, description=">0 时只处理前 N 页"),
        dpi: int = Form(150, ge=72, le=400, description="OCR 渲染精度，小字/竖排调到 200~300"),
        min_chars: int = Form(30, ge=0, le=500, description="低于该有效字符数即判定为扫描页"),
    ) -> dict[str, Any]:
        """上传 PDF 入库。

        扫描件没有文本层，**不 OCR 就会入库一堆空白卡片而且毫无报错**，
        所以这里默认 ocr=auto：按页判有效字符数，不够的页自动走 OCR。
        OCR 依赖缺失时返回 503 并说明装什么，绝不静默跳过。
        """
        name = (source or "").strip() or Path(file.filename or "未命名.pdf").name
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="上传的文件是空的")
        if len(data) > MAX_PDF_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"文件 {len(data) / 1048576:.1f}MB 超过上限 {MAX_PDF_BYTES // 1048576}MB",
            )

        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(data)
                tmp_path = Path(tmp.name)

            doc = extract_pdf(
                tmp_path,
                source=name,
                ocr=ocr,
                min_chars=min_chars,
                dpi=dpi,
                max_pages=max_pages,
            )
            chunks = pdf_chunks(doc, size=size, overlap=overlap, source=name)
        except OcrUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except (PdfError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"{name} 解析失败：{exc}") from exc
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

        if not chunks:
            # 把诊断信息一起返回：用户才知道是"没文字"还是"文字在图片里"
            raise HTTPException(
                status_code=422,
                detail={
                    "message": f"{name} 没有可入库的文本（{doc.total_pages} 页）",
                    "pages": doc.total_pages,
                    "empty_pages": doc.empty_pages,
                    "warnings": doc.warnings,
                    "hint": "如果是扫描件，请确认 ocr=auto 且已安装 rapidocr-onnxruntime",
                },
            )

        replaced = container.store.delete_by_source_prefix(name) if replace else 0
        if replaced:
            container.retriever.rebuild()
        added = container.retriever.index(chunks)
        return {
            "source": name,
            "indexed": added,
            "replaced": replaced,
            "total": len(container.retriever),
            **{k: v for k, v in doc.stats().items() if k != "source"},
        }

    @app.post("/knowledge/index-audio")
    async def knowledge_index_audio(
        file: UploadFile = File(..., description="录音文件（wav/mp3/m4a/flac/aac/ogg）"),
        source: str = Form("", description="来源标识，默认取文件名"),
        size: int = Form(300, ge=80, le=1200, description="单张卡片字数"),
        gap_ms: int = Form(1200, ge=0, le=10000, description="停顿超过该毫秒数即换卡片"),
        speaker: bool = Form(False, description="启用说话人分离（更慢，首次要下 cam++ 模型）"),
        max_seconds: int = Form(0, ge=0, le=7200, description=">0 时只转写前 N 秒（长录音试跑）"),
        replace: bool = Form(True, description="按同名来源覆盖旧卡片"),
        transcript: str = Form("", description="人工校对稿；传了就直接用文字，不再跑识别"),
    ) -> dict[str, Any]:
        """上传录音入库（口述史转写）。

        两条路都通：
        1. 直接转写 —— 适合普通话清楚、赶时间；
        2. 传 `transcript` 校对稿 —— 方言口述史的正确姿势：先转写导出、人工改错字、
           再回灌。传了校对稿就完全不跑识别，也就不需要模型和网络。

        识别不出内容时返回 422 并带告警，**绝不入库零张卡片还报成功**。
        """
        name = (source or "").strip() or Path(file.filename or "未命名音频").name
        suffix = Path(file.filename or "").suffix.lower()
        has_transcript = bool((transcript or "").strip())

        # 只有真正要跑识别时才校验音频格式：传校对稿时文件内容根本不会被读取，
        # 用一个 .txt 占位是正当用法，不该被 415 挡住。
        if not has_transcript and suffix and suffix not in AUDIO_EXTS:
            raise HTTPException(
                status_code=415,
                detail=f"不支持的音频格式 {suffix}，支持：{'、'.join(sorted(AUDIO_EXTS))}",
            )

        tmp_path: Path | None = None
        try:
            if has_transcript:
                # 人工校对稿优先：根本不落盘音频，也不需要 ASR 依赖
                tr = parse_transcript(transcript, source=name, engine="human")
                if not tr.segments:
                    raise HTTPException(status_code=422, detail="校对稿里没有解析出任何句子")
                chunks = transcript_chunks(tr, size=size, gap_ms=gap_ms, source=name)
            else:
                data = await file.read()
                if not data:
                    raise HTTPException(status_code=400, detail="上传的文件是空的")
                if len(data) > MAX_AUDIO_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"文件 {len(data) / 1048576:.0f}MB 超过上限 {MAX_AUDIO_BYTES // 1048576}MB",
                    )
                with tempfile.NamedTemporaryFile(suffix=suffix or ".wav", delete=False) as tmp:
                    tmp.write(data)
                    tmp_path = Path(tmp.name)
                tr = transcribe_audio(
                    tmp_path,
                    source=name,
                    engine=container.get_asr(speaker),
                    max_seconds=max_seconds,
                )
                if not tr.segments:
                    raise HTTPException(
                        status_code=422,
                        detail={
                            "message": f"{name} 没有识别出语音内容",
                            "duration": tr.stats()["duration"],
                            "warnings": tr.warnings,
                            "hint": "确认录音不是静音/纯噪声；方言很重时建议转写后用校对稿回灌",
                        },
                    )
                chunks = transcript_chunks(tr, size=size, gap_ms=gap_ms, source=name)
        except AsrUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except AudioError as exc:
            raise HTTPException(status_code=422, detail=f"{name} 处理失败：{exc}") from exc
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

        if not chunks:
            raise HTTPException(status_code=422, detail=f"{name} 转写成功但没有可入库的内容")

        replaced = container.store.delete_by_source_prefix(name) if replace else 0
        if replaced:
            container.retriever.rebuild()
        added = container.retriever.index(chunks)
        return {
            "source": name,
            "indexed": added,
            "replaced": replaced,
            "total": len(container.retriever),
            **{k: v for k, v in tr.stats().items() if k != "source"},
            "chunks_preview": [
                {"at": c.meta["start_ms"], "text": c.text[:40]} for c in chunks[:3]
            ],
        }

    @app.post("/knowledge/purge")
    async def knowledge_purge(body: PurgeIn) -> dict[str, Any]:
        """按来源前缀删卡片。删除后必须重建 BM25 索引，否则文档会与 token 错位。"""
        deleted = container.store.delete_by_source_prefix(body.source_prefix)
        container.retriever.rebuild()
        return {"deleted": deleted, "total": len(container.retriever)}

    @app.post("/session")
    async def create_session() -> dict[str, Any]:
        session = Session.create(pipeline=container.pipeline, tts=container.tts)
        container.sessions[session.id] = session
        return {"session_id": session.id, "state": session.state.value}

    def _get_session(session_id: str) -> Session:
        session = container.sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"会话不存在: {session_id}")
        return session

    @app.post("/session/{session_id}/say")
    async def say(session_id: str, body: SayIn):
        from fastapi.responses import StreamingResponse

        session = _get_session(session_id)

        async def event_stream():
            async for event in session.say(body.text):
                yield json.dumps(event, ensure_ascii=False) + "\n"

        return StreamingResponse(event_stream(), media_type="application/x-ndjson")

    @app.post("/session/{session_id}/interrupt")
    async def interrupt(session_id: str) -> dict[str, Any]:
        session = _get_session(session_id)
        session.interrupt()
        return {"ok": True, "interrupts": session.interrupt_count}

    @app.post("/knowledge/seed")
    async def knowledge_seed(body: SeedIn) -> dict[str, Any]:
        """把内置示例语料灌进线上知识库，供演示与人工测试。

        语料只维护在 data/corpus 一处，页面上不再硬编码一份副本，
        避免「页面里的资料」和「评测用的资料」悄悄跑偏。
        """
        from .eval import build_corpus

        path = cfg.eval_corpus_path
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"示例语料缺失：{path}")
        chunks = build_corpus(path)
        if body.reset:
            container.store.clear()
            container.retriever.rebuild()
        indexed = container.retriever.index(chunks)
        return {"indexed": indexed, "total": len(container.retriever)}

    @app.post("/eval/run")
    async def eval_run() -> dict[str, Any]:
        """跑一遍内置评测集，返回 hit@1 / MRR / 分级命中率。

        用的是独立的内存索引，不碰线上知识库，随时可跑。
        """
        from .eval import build_corpus, evaluate, read_jsonl

        corpus_path = cfg.eval_corpus_path
        qa_path = cfg.eval_qa_path
        if not corpus_path.exists() or not qa_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"评测集缺失：{corpus_path} 或 {qa_path}",
            )
        chunks = build_corpus(corpus_path)
        qa = read_jsonl(qa_path)
        return evaluate(
            container.embedder,
            chunks,
            qa,
            keyword_weight=container.keyword_weight,
            top_k=cfg.retrieve_top_k,
        )

    # 联调演示台（静态页面，无构建步骤）。目录不存在时跳过，不影响接口可用性。
    web_dir = Path(__file__).resolve().parents[2] / "web"
    if web_dir.is_dir():

        @app.get("/", include_in_schema=False)
        async def root() -> RedirectResponse:
            return RedirectResponse(url="/demo/")

        app.mount("/demo", StaticFiles(directory=str(web_dir), html=True), name="demo")

    return app


app = create_app()


def main() -> None:  # pragma: no cover - 启动入口
    import uvicorn

    cfg = get_settings()
    uvicorn.run("feiyi.api:app", host=cfg.host, port=cfg.port, reload=False)
