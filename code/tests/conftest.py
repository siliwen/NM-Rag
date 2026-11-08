import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# 测试环境强制使用离线组件，避免误连外部服务
os.environ.setdefault("EMBEDDER", "hash")
os.environ.setdefault("LLM_API_KEY", "")
# 强制内存向量库：测试不得写真实 data/index，否则用例之间互相污染。
# 需要测持久化的用例会显式传 vector_store="sqlite" + tmp_path。
os.environ["VECTOR_STORE"] = "memory"
# 测试不得连真实 TTS 后端：.env 里配了 9880 时，启动预热会真发合成请求
# （后端没起就白等超时，起了又让测试依赖本地进程）。一律走 StubTTS。
os.environ["TTS_BASE_URL"] = ""
