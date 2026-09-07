"""File and directory utility functions."""

import os
import shutil
import time
from pathlib import Path
from typing import Optional


def rmtree_robust(path: str, retries: int = 4, delay: float = 2.0) -> bool:
    """Delete a directory tree, tolerating Windows file locks.

    杀毒软件会以「读共享、拒删除/改名」锁住新写入的文件(实测可达数十秒):
    - 失败时短退避重试,让锁自行释放;
    - 顺带清理只读位(防御性,git 仓库一般没有);
    - 最终仍失败返回 False 且不做静默部分删除——由调用方决定报错或放弃,
      否则残留半删目录会让后续 git clone 报出令人困惑的 "destination not empty"。
    路径不存在视为成功(幂等)。
    """
    if not os.path.exists(path) and not os.path.islink(path):
        return True

    def _on_error(func, name, exc):
        try:
            os.chmod(name, 0o777)
            func(name)
        except Exception:  # noqa: BLE001
            pass

    for attempt in range(retries):
        try:
            shutil.rmtree(path, onerror=_on_error)
            return True
        except OSError:
            if attempt < retries - 1:
                time.sleep(delay)
    return not os.path.exists(path) and not os.path.islink(path)


def copy_with_retry(src: str, dst: str, *, retries: int = 5, delay: float = 1.0) -> str:
    """copy2 with bounded retry on OSError (locks are usually transient)."""
    for attempt in range(retries):
        try:
            return shutil.copy2(src, dst)
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(delay)
    raise OSError(f"copy failed after {retries} attempts: {src}")  # pragma: no cover


def safe_remove(path: str):
    """Safely remove a file or directory."""
    if os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


def find_files(root: str, extensions: Optional[list[str]] = None, max_depth: int = 10) -> list[str]:
    """Find files in a directory tree, optionally filtered by extension."""
    results = []
    root_path = Path(root)
    if not root_path.exists():
        return results

    for file_path in root_path.rglob("*"):
        if file_path.is_file():
            if max_depth and len(file_path.relative_to(root_path).parts) > max_depth:
                continue
            if extensions is None:
                results.append(str(file_path))
            elif file_path.suffix in extensions:
                results.append(str(file_path))
    return results


def find_python_files(root: str, max_depth: int = 10) -> list[str]:
    """Find all Python files in a directory tree."""
    return find_files(root, [".py"], max_depth)


def read_file(path: str) -> Optional[str]:
    """Read a file and return its content."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return None


def get_file_size(path: str) -> int:
    """Get file size in bytes."""
    try:
        return os.path.getsize(path)
    except Exception:
        return 0


def is_binary_file(path: str) -> bool:
    """Check if a file is likely binary."""
    try:
        with open(path, "rb") as f:
            chunk = f.read(8192)
            return b"\0" in chunk
    except Exception:
        return True