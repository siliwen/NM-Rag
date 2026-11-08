"""配置：优先读环境变量，其次读 .env（不引入 python-dotenv 依赖）。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# 默认数据目录：项目根（code/）下的 data，避免跟着启动目录漂移
_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_dotenv(path: str | Path = ".env") -> None:
    """极简 .env 解析：KEY=VALUE，忽略注释与空行，不覆盖已存在的环境变量。"""
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _get(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _get_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _get_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    """全部可配置项。默认值保证「不填任何密钥也能启动并跑测试」。"""

    llm_base_url: str = field(default_factory=lambda: _get("LLM_BASE_URL", "https://api.openai.com/v1"))
    llm_api_key: str = field(default_factory=lambda: _get("LLM_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: _get("LLM_MODEL", "gpt-4o-mini"))
    llm_timeout: int = field(default_factory=lambda: _get_int("LLM_TIMEOUT", 60))
    # 普通问答的输出上限。清单型问题（「列出所有…」）会自动放宽到 6000，
    # 800 只能写出约 1300 汉字，列到第 4 个项目就被截断，看起来像「只找到 4 个」。
    llm_max_tokens: int = field(default_factory=lambda: _get_int("LLM_MAX_TOKENS", 1200))

    # 本地语音合成服务；为空时保留静音桩，避免开发环境被强制依赖 GPU 服务。
    tts_base_url: str = field(default_factory=lambda: _get("TTS_BASE_URL", ""))
    tts_voice_id: str = field(default_factory=lambda: _get("TTS_VOICE_ID", ""))
    tts_timeout: int = field(default_factory=lambda: _get_int("TTS_TIMEOUT", 60))

    embedder: str = field(default_factory=lambda: _get("EMBEDDER", "hash"))
    embedder_device: str = field(default_factory=lambda: _get("EMBEDDER_DEVICE", ""))
    hf_endpoint: str = field(
        default_factory=lambda: _get("HF_ENDPOINT", "https://hf-mirror.com")
    )
    vector_store: str = field(default_factory=lambda: _get("VECTOR_STORE", "sqlite"))
    data_dir: str = field(default_factory=lambda: _get("DATA_DIR", str(_DEFAULT_DATA_DIR)))
    retrieve_top_k: int = field(default_factory=lambda: _get_int("RETRIEVE_TOP_K", 3))
    retrieve_candidates: int = field(default_factory=lambda: _get_int("RETRIEVE_CANDIDATES", 10))
    # 拼 prompt 时的上下文预算（字符）。1500 只装得下约 6 张卡，
    # 一顿回答 4 段以上就会被截掉尾巴，模型只能拿前半段作答。
    max_context_chars: int = field(default_factory=lambda: _get_int("MAX_CONTEXT_CHARS", 3000))
    # -1 = 自动：有语义向量时偏向量（0.2），只有 hash 桩时只能靠关键词（0.6）。
    # 实测依据：bge-small-zh 下 w=0 的 hit@1 为 88.5%，w=0.6 只有 80.8%。
    keyword_weight: float = field(default_factory=lambda: _get_float("KEYWORD_WEIGHT", -1.0))

    avatar_base_url: str = field(default_factory=lambda: _get("AVATAR_BASE_URL", "http://127.0.0.1:8010"))
    avatar_id: str = field(default_factory=lambda: _get("AVATAR_AVATAR_ID", "wav2lip256_avatar1"))
    avatar_timeout: int = field(default_factory=lambda: _get_int("AVATAR_TIMEOUT", 15))

    host: str = field(default_factory=lambda: _get("HOST", "0.0.0.0"))
    port: str = field(default_factory=lambda: _get_int("PORT", 8020))

    eval_corpus: str = field(default_factory=lambda: _get("EVAL_CORPUS", ""))
    eval_qa: str = field(default_factory=lambda: _get("EVAL_QA", ""))

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_api_key)

    def resolve_keyword_weight(self, has_semantic_embedder: bool) -> float:
        """关键词权重：显式配置优先，否则按向量能力自动取值。

        为什么自动：hash 桩没有语义能力，「影子戏」和「皮影戏」在它眼里毫无关系，
        此时必须靠关键词兜底；一旦接上真正的语义模型，关键词反而会干扰语义排序
        （实测 hit@1 从 88.5% 掉到 80.8%）。
        """
        if self.keyword_weight >= 0:
            return self.keyword_weight
        return 0.2 if has_semantic_embedder else 0.6

    @property
    def eval_corpus_path(self) -> Path:
        return (
            Path(self.eval_corpus)
            if self.eval_corpus
            else Path(self.data_dir) / "corpus" / "feiyi_demo.jsonl"
        )

    @property
    def eval_qa_path(self) -> Path:
        return (
            Path(self.eval_qa)
            if self.eval_qa
            else Path(self.data_dir) / "eval" / "qa_set.jsonl"
        )

    @property
    def index_db_path(self) -> Path:
        """SQLite 向量库落盘位置：<data_dir>/index/knowledge.db"""
        return Path(self.data_dir) / "index" / "knowledge.db"


def get_settings() -> Settings:
    return Settings()
