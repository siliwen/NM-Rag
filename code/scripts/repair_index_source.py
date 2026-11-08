"""按来源重建切片：修复「项目标题被切到上一张卡末尾」造成的项目名丢失。

背景：早期切块把「项目名 + 角色 + 起止时间」这类标题并进了上一张卡的末尾，
描述项目的卡片因此没有项目名。检索到那种卡片时，大模型只能自己编一个项目名——
程序不报错，输出却是假的（NO-006 同类）。新切块规则会把标题剥成独立片段并与
它自己的描述同卡，但已经入库的卡片不会自动变化，必须重切一遍。

用法（服务不必启动，直接操作 SQLite）：
    python scripts/repair_index_source.py "宋伟-项目经理.pdf" --dry-run   # 先看会变成什么样
    python scripts/repair_index_source.py "宋伟-项目经理.pdf"             # 真正重建

有原始文件时优先重新上传（`/knowledge/index-pdf`），那条路径最干净；
本脚本用于原始文件已丢失、只剩库内卡片的情况。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from feiyi.config import get_settings, load_dotenv  # noqa: E402
from feiyi.rag.chunking import chunk_text  # noqa: E402
from feiyi.rag.embedder import build_embedder  # noqa: E402
from feiyi.rag.retriever import HybridRetriever  # noqa: E402
from feiyi.rag.store import SqliteVectorStore  # noqa: E402

load_dotenv(ROOT / ".env")


def merge_cards(texts: list[str], max_probe: int = 80) -> str:
    """把连续卡片拼回全文，顺手去掉相邻卡之间的重叠。

    两个细节：
    - 切块时下一张卡的头部会重复上一张卡的尾部（overlap），不去掉的话重切后重复会累积
    - **不能插换行**：切块可能把词硬切成两半（「Vibe Coding」→「Vi」+「be Coding」），
      插换行会把词永久拆开；原文本来就是连续的，直接接上才对
    """
    merged = texts[0]
    for cur in texts[1:]:
        keep = 0
        for size in range(min(len(merged), len(cur), max_probe), 3, -1):
            if merged.endswith(cur[:size]):
                keep = size
                break
        merged += cur[keep:]
    return merged


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="要重建的来源名，例如 宋伟-项目经理.pdf")
    parser.add_argument("--dry-run", action="store_true", help="只打印前后对比，不写库")
    parser.add_argument("--size", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    db_path = Path(settings.data_dir) / "index" / "knowledge.db"
    if not db_path.exists():
        print(f"找不到索引库：{db_path}")
        return 1

    store = SqliteVectorStore(db_path)
    old = [c for c in store.chunks() if c.source == args.source]
    if not old:
        print(f"来源「{args.source}」在库里没有卡片。现有来源：")
        for src in sorted({c.source for c in store.chunks()}):
            print(f"  - {src}")
        return 1

    full = merge_cards([c.text.strip() for c in old])
    new = chunk_text(full, size=args.size, overlap=args.overlap, source=args.source)

    print(f"来源：{args.source}")
    print(f"原文拼接：{len(full)} 字")
    print(f"卡片数：{len(old)} -> {len(new)}")
    if new:
        avg_old = sum(len(c.text) for c in old) / len(old)
        avg_new = sum(len(c.text) for c in new) / len(new)
        print(f"平均长度：{avg_old:.0f} -> {avg_new:.0f} 字")
    print("\n重建后的前 3 张：")
    for c in new[:3]:
        print("  -", c.text[:70].replace("\n", " "))

    if args.dry_run:
        print("\n[dry-run] 未写库。确认无误后去掉 --dry-run 再跑一次。")
        store.close()
        return 0

    if not args.no_backup:
        backup = db_path.with_suffix(f".db.bak-{datetime.now():%Y%m%d-%H%M%S}")
        shutil.copy2(db_path, backup)
        print(f"\n已备份：{backup}")

    removed = store.delete_by_source_prefix(args.source)
    print(f"已删除旧卡片 {removed} 张")

    embedder = build_embedder(settings.embedder, settings.embedder_device or None)
    retriever = HybridRetriever(
        embedder=embedder,
        store=store,
        keyword_weight=settings.resolve_keyword_weight(
            has_semantic_embedder=hasattr(embedder, "model_id")
        ),
    )
    added = retriever.index(new)
    print(f"已写入新卡片 {added} 张，索引总数 {len(store)}")
    store.close()
    print("\n完成。请重启编排层（scripts/serve.py）让新索引生效。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
