"""Disk-based cache for LLM calls, paper PDFs, and repository clones.

Uses a simple JSON + file-based scheme under <project root>/.oscar_cache/
(git-ignored, project-local — 非 ~/.oscar_cache)。
LLM requests are cached by (prompt_hash + model), papers by arxiv_id,
repos by URL hash.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Optional

from oscar.config import config
from oscar.utils.file_utils import rmtree_robust, copy_with_retry

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(os.path.dirname(__file__)).parent.parent / ".oscar_cache"
_LLM_DIR = _CACHE_DIR / "llm"
_PAPER_DIR = _CACHE_DIR / "papers"
_REPO_DIR = _CACHE_DIR / "repos"
_VECTOR_DIR = _CACHE_DIR / "vectors"

_TTL_DEFAULT = config.cache.ttl_seconds  # 7 天(默认;经 config.yaml cache.ttl_seconds 调整)


def _ensure_dirs():
    for d in [_LLM_DIR, _PAPER_DIR, _REPO_DIR, _VECTOR_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# LLM call cache
# ---------------------------------------------------------------------------

def _llm_key(messages: list[dict], model: str, temperature: float, salt: str = "") -> str:
    """Generate a deterministic hash key for an LLM request."""
    raw = json.dumps({"messages": messages, "model": model, "temperature": temperature, "salt": salt}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def llm_cache_get(messages: list[dict], model: str, temperature: float = 0.0, salt: str = "") -> Optional[dict]:
    """Return cached LLM response if available and not expired."""
    _ensure_dirs()
    key = _llm_key(messages, model, temperature, salt)
    path = _LLM_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        # 写用 utf-8;读必须显式 utf-8 —— 默认 locale 编码(中文 Windows 为
        # cp936)解码含非 ASCII 的响应(如代码片段里的中文注释)会抛错或被
        # 静默解读成乱码,导致缓存「存在却永远 miss」→ 每次运行重调真实 API。
        data = json.loads(path.read_text(encoding="utf-8"))
        if time.time() - data["cached_at"] > _TTL_DEFAULT:
            path.unlink(missing_ok=True)
            return None
        return data["response"]
    except Exception:
        return None


def llm_cache_set(messages: list[dict], model: str, temperature: float, response: dict, salt: str = ""):
    """Cache an LLM response."""
    _ensure_dirs()
    key = _llm_key(messages, model, temperature, salt)
    path = _LLM_DIR / f"{key}.json"
    path.write_text(json.dumps({
        "key": key,
        "model": model,
        "cached_at": time.time(),
        "response": response,
    }, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Paper PDF cache  (by arxiv_id)
# ---------------------------------------------------------------------------

def paper_cache_get(arxiv_id: str) -> Optional[str]:
    """Return cached PDF path for a paper, or None."""
    _ensure_dirs()
    path = _PAPER_DIR / f"{arxiv_id}.pdf"
    if path.exists():
        return str(path)
    # Also check for extracted text
    txt_path = _PAPER_DIR / f"{arxiv_id}.txt"
    if txt_path.exists():
        return str(txt_path)
    return None


def paper_cache_set(arxiv_id: str, source_path: str):
    """Cache a paper PDF or text file by arxiv_id."""
    _ensure_dirs()
    ext = os.path.splitext(source_path)[1] or ".pdf"
    dest = _PAPER_DIR / f"{arxiv_id}{ext}"
    shutil.copy2(source_path, dest)


def paper_cache_has_text(arxiv_id: str) -> Optional[str]:
    """Return cached extracted text path, or None."""
    path = _PAPER_DIR / f"{arxiv_id}.txt"
    if path.exists():
        return str(path)
    return None


def paper_cache_set_text(arxiv_id: str, text: str):
    """Cache extracted paper text."""
    _ensure_dirs()
    path = _PAPER_DIR / f"{arxiv_id}.txt"
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Repository clone cache (by repo URL hash)
# ---------------------------------------------------------------------------

REPO_CACHE_TTL = _TTL_DEFAULT  # 仓库克隆缓存有效期(7 天),供 loader 原位复用判断


def _repo_key(repo_url: str) -> str:
    return hashlib.sha256(repo_url.encode()).hexdigest()[:16]


def repo_cache_get(repo_url: str) -> Optional[str]:
    """Return cached repo clone path, or None.

    有效性校验:目录含 .git **且** 存在提升完成标记 <key>.ok——防止把上次
    提升失败留下的半成品目录(有 .git 但缺文件)误判为可用缓存。
    """
    _ensure_dirs()
    key = _repo_key(repo_url)
    path = _REPO_DIR / key
    marker = _REPO_DIR / f"{key}.ok"
    if path.exists() and marker.exists() and (path / ".git").exists():
        # Check staleness: 7 days
        age = time.time() - path.stat().st_mtime
        if age < _TTL_DEFAULT:
            return str(path)
        # Stale → remove (含标记)
        rmtree_robust(str(path))
        marker.unlink(missing_ok=True)
    elif path.exists():
        # 半成品/损坏缓存 → 清除
        rmtree_robust(str(path))
        marker.unlink(missing_ok=True)
    return None


def repo_cache_set(repo_url: str, clone_path: str) -> Optional[str]:
    """Promote an in-place clone into the cache. Never raises; returns the
    promoted cache path, or None when promotion fails (clone is left in place).

    Windows 上的关键细节:杀毒软件实时扫描会「读共享、拒删改名」地锁住
    新克隆的 pack 文件(实测可长达数十秒),rename 会短暂 WinError 5。
    - 先尝试 os.rename(同卷 atomic,命中即去重、保持单份拷贝)。注意必须
      用 os.rename 直连:shutil.move 在 rename 失败后会静默回退到
      copytree+rmtree,锁住文件时复制一半才抛错,留下带 .git 的半成品;
    - 短退避重试仍失败 → 退化为显式整树复制(锁定文件允许共享读,复制
      copy_with_retry 兜底),成功后尽力删除原位克隆;
    - 提升成功的唯一凭证是写入 <key>.ok 标记;任何失败路径都不写标记,
      半成品目录会被 repo_cache_get 识别并清除。
    - 全部失败 → 返回 None,克隆保留在 clone_path 原位,本次不缓存
      (审计照常进行,只是下次需重新克隆)。
    """
    _ensure_dirs()
    if not os.path.isdir(clone_path):
        return None  # 源目录不存在/已被提升过 → 无需处理
    key = _repo_key(repo_url)
    dest = _REPO_DIR / key
    marker = _REPO_DIR / f"{key}.ok"
    marker.unlink(missing_ok=True)

    # 清理可能残留的旧缓存/半成品(旧文件不会被杀软锁定,正常可删;
    # 若为新写文件且仍被锁,rmtree_robust 带退避重试)
    if dest.exists() or os.path.islink(str(dest)):
        if not rmtree_robust(str(dest)):
            logger.warning("Cannot clear stale cache entry %s; skipping promotion", dest)
            return None

    # 1) rename 快路径(去重,保留单份)。被锁时短退避重试。
    for delay in (0.0, 1.0, 2.0):
        try:
            os.rename(clone_path, str(dest))
            marker.touch()
            return str(dest)
        except OSError:
            if delay:
                time.sleep(delay)

    # 2) 复制回退:锁定文件允许共享读,整树复制不受影响
    try:
        shutil.copytree(clone_path, str(dest), copy_function=copy_with_retry)
        marker.touch()
    except Exception:
        # 复制中途失败(dest 为半成品,无标记 → 不会命中校验)
        rmtree_robust(str(dest))
        logger.warning("Cache promotion (copy) failed for %s", clone_path, exc_info=True)
        return None

    # 复制成功后尽力删除原位克隆(锁此时多半已释放);失败留待用户/下次清理
    if not rmtree_robust(clone_path):
        logger.warning(
            "In-place clone left at %s (locked files); it is a duplicate of the "
            "cache copy and can be deleted manually",
            clone_path,
        )
    return str(dest)


# ---------------------------------------------------------------------------
# Vector store persistence (by arxiv_id)
# ---------------------------------------------------------------------------

def vector_cache_get(arxiv_id: str) -> Optional[dict]:
    """Return cached vector store data for a paper."""
    _ensure_dirs()
    path = _VECTOR_DIR / f"{arxiv_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def vector_cache_set(arxiv_id: str, data: dict):
    """Cache vector store data for a paper."""
    _ensure_dirs()
    path = _VECTOR_DIR / f"{arxiv_id}.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Code vector store persistence (by repo URL hash)
# ---------------------------------------------------------------------------

CODE_VECTOR_DIR = _VECTOR_DIR / "code"  # .oscar_cache/vectors/code/{key}.faiss/.json


def repo_key(repo_url: str) -> str:
    """Deterministic repo URL hash key (公开:code vector 持久化等复用)."""
    return _repo_key(repo_url)


def atomic_replace(tmp_path, dest_path, retries: int = 3, delay: float = 1.0) -> bool:
    """Atomically move tmp file over dest with retries.

    Windows 上新建文件会被杀毒软件短暂锁定(共享读、拒改名),rename 可能
    WinError 5;短退避重试后仍失败则返回 False(调用方负责清理 tmp / 下次重建)。
    """
    for attempt in range(retries):
        try:
            os.replace(tmp_path, dest_path)
            return True
        except OSError:
            if attempt < retries - 1:
                time.sleep(delay)
    return False


def code_vector_paths(repo_url: str) -> tuple[Path, Path]:
    """Return (faiss index path, meta json path) for a repo."""
    key = repo_key(repo_url)
    return CODE_VECTOR_DIR / f"{key}.faiss", CODE_VECTOR_DIR / f"{key}.json"


def code_vector_meta_read(repo_url: str) -> Optional[dict]:
    """Read code vector index meta; returns None on missing/corrupt."""
    _, meta_path = code_vector_paths(repo_url)
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def code_vector_meta_write(repo_url: str, meta: dict) -> bool:
    """Persist code vector meta json. Written last as completion marker."""
    _ensure_dirs()
    CODE_VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    _, meta_path = code_vector_paths(repo_url)
    tmp = meta_path.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        return atomic_replace(tmp, meta_path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return False


def code_vector_delete(repo_url: str):
    """Remove persisted code vector index for a repo (best effort)."""
    index_path, meta_path = code_vector_paths(repo_url)
    for p in (index_path, meta_path, meta_path.with_suffix(".json.tmp")):
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def clear_cache(older_than_days: int = 7):
    """Remove all cached entries older than the specified age."""
    cutoff = time.time() - older_than_days * 86400
    for root, dirs, files in os.walk(str(_CACHE_DIR)):
        for f in files:
            path = Path(root) / f
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
            except Exception:
                pass
    # Remove empty dirs
    for dirpath, dirs, files in os.walk(str(_CACHE_DIR), topdown=False):
        if not dirs and not files and dirpath != str(_CACHE_DIR):
            try:
                os.rmdir(dirpath)
            except Exception:
                pass


def cache_stats() -> dict:
    """Return cache statistics."""
    _ensure_dirs()
    return {
        "llm": len(list(_LLM_DIR.glob("*.json"))),
        "papers": len(list(_PAPER_DIR.glob("*"))),
        "repos": len(list(_REPO_DIR.glob("*"))),
        "vectors": len(list(_VECTOR_DIR.glob("*.json"))),
        "cache_dir": str(_CACHE_DIR),
    }