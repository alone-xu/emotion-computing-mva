"""把仓库打包成可直接上传 GitHub 的 zip。

包含：文档、源码、测试、脚本、示例产物、配置。
排除：运行时数据（data/，含 ~1 GB 数据集）、内部记忆（.workbuddy/）、缓存。

用法：
    python scripts/build_release_zip.py
产物：
    dist/emotion-computing-mva-audit.zip
"""

from __future__ import annotations

import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
ZIP_NAME = "emotion-computing-mva-audit.zip"
TOP = "emotion-computing"  # zip 内的根目录名

# 目录级白名单（整个目录递归包含）
INCLUDE_DIRS = ("docs", "src", "tests", "scripts", "examples")
# 单文件白名单
INCLUDE_FILES = (
    ".gitignore",
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "models/README.md",
    "data/.gitkeep",
)

# 任何路径包含这些片段则跳过（缓存/构建产物）
EXCLUDE_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".egg-info",
    ".mypy_cache",
    ".ruff_cache",
    ".workbuddy",
    "dist",
    "build",
    ".venv",
    "venv",
}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".pyd", ".so", ".log"}


def should_skip(rel: Path) -> bool:
    parts = set(rel.parts)
    if parts & EXCLUDE_PARTS:
        return True
    if any(p.endswith(".egg-info") for p in rel.parts):
        return True
    if rel.suffix.lower() in EXCLUDE_SUFFIXES:
        return True
    if rel.name in {".DS_Store", "Thumbs.db"}:
        return True
    return False


def collect() -> list[Path]:
    files: list[Path] = []
    for name in INCLUDE_FILES:
        p = ROOT / name
        if p.is_file():
            files.append(p)
    for d in INCLUDE_DIRS:
        base = ROOT / d
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            if should_skip(p.relative_to(ROOT)):
                continue
            files.append(p)
    return sorted(set(files))


def main() -> int:
    DIST.mkdir(parents=True, exist_ok=True)
    out = DIST / ZIP_NAME
    files = collect()
    if not files:
        print("没有可打包的文件")
        return 1

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for f in files:
            rel = f.relative_to(ROOT)
            zf.write(f, arcname=f"{TOP}/{rel.as_posix()}")

    total_uncompressed = sum(f.stat().st_size for f in files)
    size = out.stat().st_size
    print(f"[OK] {out}")
    print(f"     文件数：{len(files)}")
    print(f"     原始大小：{total_uncompressed / 1024:.1f} KB")
    print(f"     压缩后：{size / 1024:.1f} KB")
    print()
    print("目录树（zip 内）：")
    from collections import Counter
    top_dirs: Counter[str] = Counter()
    for f in files:
        rel = f.relative_to(ROOT)
        key = rel.parts[0] if len(rel.parts) > 1 else "(root)"
        top_dirs[key] += 1
    for k, v in sorted(top_dirs.items()):
        print(f"  {k:<14} {v} 个文件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
