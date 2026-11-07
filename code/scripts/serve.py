"""启动服务：自动把 src 加入路径、加载 .env，再交给 uvicorn。

用法：
    python scripts/serve.py
    python scripts/serve.py --port 8030 --reload
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from feiyi.config import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import uvicorn  # noqa: E402

from feiyi.config import get_settings  # noqa: E402


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run("feiyi.api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
