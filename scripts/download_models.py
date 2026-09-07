"""Download the local embedding models OSCAR needs into bert/.

背景:OSCAR 的代码/论文向量检索默认加载本地 Hugging Face 模型(路径按
仓库内硬编码为 <项目根>/bert/…,见 oscar/mapping/code_vector_store.py 与
oscar/paper/vector_store.py)。模型资产体积大(数 GB),**不随仓库分发**
(bert/ 已在 .gitignore)——首次使用前运行本脚本即可,已下载过则幂等跳过。

用法:
    python scripts/download_models.py                 # 只下 codebert-base
    python scripts/download_models.py --dry-run       # 只看计划,不联网
    python scripts/download_models.py --also-graphcodebert

说明:
- codebert-base(≈1.9 GB,必需):代码块 embedding,FAISS 向量检索。
- graphcodebert-base(≈1.7 GB,可选):仓库里无引用;仅在显式
  --also-graphcodebert 时下载。
- 下载失败(网络/墙)时,重试前可设置国内镜像:
      set HF_ENDPOINT=https://hf-mirror.com        (Windows cmd)
      export HF_ENDPOINT=https://hf-mirror.com     (bash)
- 完成后如需手动验证:bert/codebert-base/model.safetensors 应存在。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_MODELS = {
    # local_dir 目录名 -> (HF repo id, 体积提示, 是否必需)
    "codebert-base": ("microsoft/codebert-base", "~1.9 GB", True),
    "graphcodebert-base": ("microsoft/graphcodebert-base", "~1.7 GB", False),
}
# 任意一个权重文件存在即视为该模型已下载完成(snapshot_download 幂等)
_WEIGHT_MARKERS = (
    "model.safetensors",
    "pytorch_model.bin",
    "tf_model.h5",
    "flax_model.msgpack",
)


def _already_downloaded(local_dir: Path) -> bool:
    return local_dir.is_dir() and any((local_dir / m).exists() for m in _WEIGHT_MARKERS)


def _plan(include_graphcodebert: bool) -> list[tuple[str, str, str, bool]]:
    """(local_dir 名, repo id, 体积, 已下载) 列表 — 排序稳定。"""
    out = []
    for name, (repo_id, size, required) in _MODELS.items():
        if name == "graphcodebert-base" and not include_graphcodebert:
            continue
        out.append((name, repo_id, size, _already_downloaded(_ROOT / "bert" / name)))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只打印计划并退出(不联网、不 import huggingface_hub)",
    )
    parser.add_argument(
        "--also-graphcodebert", action="store_true",
        help="额外下载 graphcodebert-base(仓库内无引用,默认不下)",
    )
    args = parser.parse_args()

    ( _ROOT / "bert" ).mkdir(parents=True, exist_ok=True)
    plan = _plan(args.also_graphcodebert)
    if not plan:
        print("Nothing to do: no model selected (add --also-graphcodebert to fetch the optional one).")
        return 0

    for name, repo_id, size, done in plan:
        # 控制台编码差异下避免非 ASCII 破折号(cp936 会显示乱码)
        state = "already present - skip" if done else "to download"
        print(f"- {name:20s} <- {repo_id} ({size}) [{state}]")

    if args.dry_run:
        print("--dry-run: no download performed.")
        return 0

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "huggingface_hub is not installed. Install it with:\n"
            "    pip install -r requirements.txt\n"
            "or: pip install huggingface-hub>=0.23",
            file=sys.stderr,
        )
        return 1

    failures = 0
    for name, repo_id, _size, done in plan:
        if done:
            continue
        dest = _ROOT / "bert" / name
        print(f"\nDownloading {repo_id} -> {dest} ...")
        try:
            snapshot_download(repo_id=repo_id, local_dir=dest)
        except Exception as exc:  # noqa: BLE001 — 网络/镜像问题统一给指引
            failures += 1
            print(
                f"Failed to download {repo_id}: {exc}\n"
                f"网络受限时可设置镜像后重试:\n"
                f'    export HF_ENDPOINT=https://hf-mirror.com   (bash)\n'
                f'    set HF_ENDPOINT=https://hf-mirror.com      (Windows cmd)',
                file=sys.stderr,
            )

    if failures:
        print(f"\n{failures} model(s) failed. See messages above.")
        return 1
    print("\nDone. OSCAR will load models from bert/ automatically.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
