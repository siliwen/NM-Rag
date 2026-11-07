"""非遗数字人讲解员 —— 会话编排 + RAG 检索服务。

设计原则：
1. 测试期不依赖 GPU / 不依赖向量模型（用 HashingEmbedder + 内存向量库）
2. 生产期只替换 embedder / store / tts / asr 的实现，编排逻辑不变
3. 唯一的外部依赖是 LLM 接口（OpenAI 兼容）
"""

__version__ = "0.1.0"
