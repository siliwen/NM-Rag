"""下载 Kokoro-82M 中文权重到本地（走 ModelScope）。

为什么不用 huggingface_hub：本机到 hf-mirror 的连接实测会在下载中途停滞
（327MB 的权重下到 270MB 后连续 4 分钟零进展，且 hf-mirror 的 API 直接不通），
modelscope.cn 上同一仓库可直连。下完的目录被 `.gitignore` 的 code/data/models/ 覆盖。

用法：
    python scripts/fetch_kokoro_model.py
    python scripts/fetch_kokoro_model.py --dest data/models/kokoro-82m-zh
"""
from __future__ import annotations

import argparse
from pathlib import Path

MODEL_ID = "hexgrad/Kokoro-82M-v1.1-zh"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = ROOT / "data" / "models" / "kokoro-82m-zh"
# 主权重 + 配置 + 音色包（每个 523KB，全下也就 2.6MB）
PATTERNS = ["config.json", "kokoro-v1_1-zh.pth", "voices/*.pt"]
REQUIRED = ["config.json", "kokoro-v1_1-zh.pth"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default=str(DEFAULT_DEST))
    parser.add_argument("--model", default=MODEL_ID)
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    from modelscope.hub.snapshot_download import snapshot_download

    print(f"[下载] {args.model} -> {dest}")
    snapshot_download(model_id=args.model, local_dir=str(dest), allow_patterns=PATTERNS)

    missing = [f for f in REQUIRED if not (dest / f).exists()]
    if missing:
        raise RuntimeError(f"下载结束但缺少必需文件：{missing}（目录 {dest}）")

    total = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
    voices = sorted(p.stem for p in (dest / "voices").glob("*.pt"))
    print(f"[完成] 共 {total / 1024 / 1024:.1f} MB，音色 {len(voices)} 个：{', '.join(voices)}")


if __name__ == "__main__":
    main()
