"""向量化。

三种实现，用途严格区分：

- HashingEmbedder：零依赖字符 n-gram 哈希。**只用于单元测试**，
  它不具备语义能力（「影子戏」和「皮影戏」在它眼里毫不相干），
  拿它测检索质量会得出错误结论。
- LocalEmbedder：sentence-transformers 加载本地 BGE 模型，CPU 可跑，
  测试期与生产期都用它。开发期用 bge-small-zh（快），生产期换 bge-m3（准）。
- BGEM3Embedder：FlagEmbedding 加载 bge-m3，走 GPU / fp16。
  仅在显存富余且需要稀疏+稠密双输出时使用。

⚠️ 换模型 = 换维度，必须清空索引重建（store 会显式报错提醒，不会静默算错）。
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Protocol, Sequence

_LATIN = re.compile(r"[a-zA-Z0-9]+")
_CJK = re.compile(r"[\u4e00-\u9fff]")

# 国内直连 huggingface.co 基本不通，默认走镜像；可用 HF_ENDPOINT 覆盖。
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"

# 别名 → HuggingFace 模型 ID
MODEL_ALIASES: dict[str, str] = {
    "bge-small-zh": "BAAI/bge-small-zh-v1.5",       # 24M 参数 / 512 维 / CPU 友好
    "bge-base-zh": "BAAI/bge-base-zh-v1.5",         # 102M / 768 维
    "bge-large-zh": "BAAI/bge-large-zh-v1.5",       # 326M / 1024 维
    "bge-m3": "BAAI/bge-m3",                        # 568M / 1024 维 / 多语言
    "bge-small": "BAAI/bge-small-zh-v1.5",
}

# BGE 中文 v1.5 系列是「非对称检索」模型：查询侧要加指令，文档侧不加。
# bge-m3 不需要指令，加了反而退化，所以按模型前缀区分。
_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


class Embedder(Protocol):
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def _features(text: str) -> list[str]:
    """中文取字符二元组 + 英文/数字整词。非遗专名（如「皮影」「秦腔」）靠二元组能稳定命中。"""
    feats = [w.lower() for w in _LATIN.findall(text or "")]
    zh = "".join(_CJK.findall(text or ""))
    feats.extend(zh[i : i + 2] for i in range(len(zh) - 1))
    return feats


class HashingEmbedder:
    """特征哈希 + 带符号累加 + L2 归一化。不是语义模型，但词面相似度可靠。

    ⚠️ 仅用于单测：它无法识别同义改写，评估检索质量必须用 LocalEmbedder。
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for feat in _features(text):
                digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "big")
                sign = 1.0 if (value >> 63) & 1 else -1.0
                vec[value % self.dim] += sign
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


def resolve_model_name(name: str) -> str:
    key = (name or "").strip()
    return MODEL_ALIASES.get(key.lower(), key)


class LocalEmbedder:
    """sentence-transformers 本地推理，CPU 可用。生产与测试共用同一套代码。"""

    def __init__(
        self,
        model_name: str = "bge-small-zh",
        device: str | None = None,
        hf_endpoint: str | None = None,
        max_length: int = 512,
    ) -> None:
        endpoint = hf_endpoint or os.environ.get("HF_ENDPOINT") or DEFAULT_HF_ENDPOINT
        os.environ["HF_ENDPOINT"] = endpoint

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - 依赖缺失路径
            raise RuntimeError(
                "需要先安装依赖：pip install sentence-transformers；"
                "仅跑单元测试时把 EMBEDDER 设为 hash"
            ) from exc

        self.model_id = resolve_model_name(model_name)
        self.query_instruction = "" if "m3" in self.model_id.lower() else _QUERY_INSTRUCTION
        try:
            self._model = SentenceTransformer(self.model_id, device=device)
        except Exception as exc:  # pragma: no cover - 网络/模型缺失路径
            raise RuntimeError(
                f"加载向量模型失败：{self.model_id}\n"
                f"镜像地址：{endpoint}\n"
                f"原始错误：{type(exc).__name__}: {exc}\n"
                "可先手动下载：huggingface-cli download "
                f"{self.model_id} --local-dir models/{self.model_id.split('/')[-1]}"
            ) from exc
        # sentence-transformers 6.x 起不接受 encode(max_length=...)，改在模型上设置
        self._model.max_seq_length = max_length
        getter = getattr(self._model, "get_embedding_dimension", None)
        self.dim = int(getter() if callable(getter) else self._model.get_sentence_embedding_dimension())

    def embed(self, texts: Sequence[str]) -> list[list[float]]:  # pragma: no cover - 需模型
        items = [t if t else " " for t in texts]
        vectors = self._model.encode(
            items,
            batch_size=8,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [[float(x) for x in vec] for vec in vectors]

    def embed_query(self, text: str) -> list[float]:  # pragma: no cover - 需模型
        """查询侧加指令（BGE 中文非对称检索约定），文档侧不加。"""
        return self.embed([self.query_instruction + (text or "")])[0]


class BGEM3Embedder:
    """FlagEmbedding 版 bge-m3（GPU + fp16）。延迟导入，缺依赖时给出可执行的提示。"""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None) -> None:
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:  # pragma: no cover - 依赖缺失路径
            raise RuntimeError(
                "需要先安装 bge-m3 依赖：pip install FlagEmbedding；"
                "或改用 EMBEDDER=bge-m3（走 sentence-transformers 加载同一模型）"
            ) from exc
        self._model = BGEM3FlagModel(model_name, use_fp16=True, device=device)
        self.dim = 1024

    def embed(self, texts: Sequence[str]) -> list[list[float]]:  # pragma: no cover - 需模型
        result = self._model.encode(list(texts), batch_size=8, max_length=512)
        return [[float(x) for x in vec] for vec in result["dense_vecs"]]


def build_embedder(name: str = "hash", device: str | None = None) -> Embedder:
    """按名字构建向量器。

    - `hash` / `stub`：测试桩，无语义能力
    - `bge-small-zh` / `bge-base-zh` / `bge-m3`：内置别名
    - `org/model` 形式：任意 HuggingFace 模型 ID

    名字写错时直接报错，不做「猜一个模型去下载」的兜底——否则一个拼写错误
    会变成几分钟的等待和一堆难以定位的报错。
    """
    key = (name or "").strip()
    low = key.lower()
    if low in {"hash", "hashing", "stub"}:
        return HashingEmbedder()
    if not key:
        raise ValueError("EMBEDDER 不能为空")
    if low in MODEL_ALIASES or "/" in key:
        return LocalEmbedder(model_name=key, device=device)
    raise ValueError(
        f"未知的 EMBEDDER: {name}，可选 hash / {' / '.join(sorted(MODEL_ALIASES))} / "
        "任意 HuggingFace 模型 ID（形如 BAAI/bge-m3）"
    )
