"""CacheManager — 通用缓存层，基于内容哈希而非平台 ID。

设计原则：
- 内容哈希：缓存 Key 仅依赖内容本身，而非来源（arXiv / IEEE / 本地）。
- 统一接口：一个 CacheManager 管理所有缓存类型（LLM、论文、代码仓库、向量）。
- 并发安全：使用 threading.Lock 保护写入，避免竞态。
- TTL 可配：每个缓存类型有独立的过期时间。

用法：
  cm = CacheManager()
  cm.get_llm(prompt, model)  → Optional[str]
  cm.set_llm(prompt, model, response)
  cm.get_paper(file_bytes, embed_version)  → Optional[dict]
  cm.set_paper(file_bytes, embed_version, data)
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Optional


class CacheManager:
    """基于内容哈希的通用缓存管理器。

    所有缓存 Key 通过 hashlib.sha256 对内容（而非来源）取哈希。
    元数据存储在 JSON 文件中，大对象（如 FAISS 索引）使用 pickle。
    """

    def __init__(self, cache_dir: str | Path | None = None):
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent.parent / ".oscar_cache"
        self._root = Path(cache_dir)
        self._lock = threading.Lock()

        for sub in ("llm", "paper", "repo", "vector"):
            (self._root / sub).mkdir(parents=True, exist_ok=True)

        # 默认 TTL（秒）
        self._ttl = {
            "llm": 86400 * 7,     # 7 天
            "paper": 86400 * 30,  # 30 天
            "repo": 86400 * 7,    # 7 天
            "vector": 86400 * 30, # 30 天
        }

    # -- 内部工具 -----------------------------------------------------------

    @staticmethod
    def _sha256(*parts: str | bytes) -> str:
        """对多个部分拼接后取 SHA-256 前缀（32 字符）。"""
        raw = "".join(
            p if isinstance(p, str) else p.decode("utf-8", errors="replace")
            for p in parts
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    def _path(self, category: str, key: str) -> Path:
        return self._root / category / f"{key}.json"

    def _get(self, category: str, key: str) -> Any | None:
        """从缓存中读取，若过期或不存在则返回 None。"""
        path = self._path(category, key)
        if not path.exists():
            return None
        try:
            with self._lock:
                data = json.loads(path.read_text(encoding="utf-8"))
            if time.time() - data["_cached_at"] > self._ttl.get(category, 86400):
                path.unlink(missing_ok=True)
                return None
            return data["_value"]
        except Exception:
            return None

    def _set(self, category: str, key: str, value: Any):
        """写入缓存。"""
        data = {"_key": key, "_cached_at": time.time(), "_value": value}
        path = self._path(category, key)
        with self._lock:
            path.write_text(json.dumps(data, ensure_ascii=False, default=str), encoding="utf-8")

    # -- LLM 缓存 ----------------------------------------------------------
    # Key: Hash(prompt_messages + model_name + temperature)

    def llm_key(self, messages: list[dict], model: str, temperature: float = 0.0) -> str:
        raw = json.dumps({"m": messages, "model": model, "t": temperature}, sort_keys=True)
        return self._sha256(raw)

    def get_llm(self, messages: list[dict], model: str, temperature: float = 0.0) -> str | None:
        return self._get("llm", self.llm_key(messages, model, temperature))

    def set_llm(self, messages: list[dict], model: str, temperature: float, response: str):
        self._set("llm", self.llm_key(messages, model, temperature), response)

    # -- 论文缓存 ----------------------------------------------------------
    # Key: Hash(pdf_bytes + embed_model_version)

    def paper_key(self, file_bytes: bytes, embed_version: str = "codebert-v1") -> str:
        return self._sha256(file_bytes, embed_version)

    def get_paper(self, file_bytes: bytes, embed_version: str = "codebert-v1") -> dict | None:
        return self._get("paper", self.paper_key(file_bytes, embed_version))

    def set_paper(self, file_bytes: bytes, embed_version: str, data: dict):
        self._set("paper", self.paper_key(file_bytes, embed_version), data)

    # -- 仓库缓存 ----------------------------------------------------------
    # Key: Hash(repo_url + commit_hash)

    def repo_key(self, repo_url: str, commit_hash: str = "") -> str:
        return self._sha256(repo_url, commit_hash or "")

    def get_repo_path(self, repo_url: str, commit_hash: str = "") -> str | None:
        key = self.repo_key(repo_url, commit_hash)
        path = self._root / "repo" / key
        if path.exists() and (path / ".git").exists():
            return str(path)
        return None

    def set_repo(self, repo_url: str, clone_path: str, commit_hash: str = ""):
        key = self.repo_key(repo_url, commit_hash)
        dest = self._root / "repo" / key
        if dest.exists():
            import shutil
            shutil.rmtree(str(dest), ignore_errors=True)
        # 软链接或复制
        os.symlink(clone_path, str(dest))

    # -- 向量缓存 ----------------------------------------------------------
    # Key: Hash(paper_id + embed_model_version)
    # 元数据存 JSON，FAISS 索引存 .faiss 文件

    def vector_key(self, paper_id: str, embed_version: str = "codebert-v1") -> str:
        return self._sha256(paper_id, embed_version)

    def get_vector_meta(self, paper_id: str, embed_version: str = "codebert-v1") -> dict | None:
        return self._get("vector", self.vector_key(paper_id, embed_version) + "_meta")

    def set_vector_meta(self, paper_id: str, embed_version: str, meta: dict):
        self._set("vector", self.vector_key(paper_id, embed_version) + "_meta", meta)

    def get_vector_index_path(self, paper_id: str, embed_version: str = "codebert-v1") -> str | None:
        key = self.vector_key(paper_id, embed_version)
        path = self._root / "vector" / f"{key}.faiss"
        return str(path) if path.exists() else None

    def set_vector_index_path(self, paper_id: str, embed_version: str, index_path: str):
        key = self.vector_key(paper_id, embed_version)
        dest = self._root / "vector" / f"{key}.faiss"
        import shutil
        shutil.copy2(index_path, str(dest))

    # -- 统计与清理 --------------------------------------------------------

    def stats(self) -> dict[str, int]:
        counts = {}
        for sub in ("llm", "paper", "repo", "vector"):
            counts[sub] = len(list((self._root / sub).glob("*")))
        counts["cache_dir"] = str(self._root)
        return counts

    def clear(self, older_than_days: int = 0):
        cutoff = time.time() - older_than_days * 86400
        for sub in ("llm", "paper", "repo", "vector"):
            for f in (self._root / sub).iterdir():
                try:
                    if f.stat().st_mtime < cutoff:
                        f.unlink() if f.is_file() else shutil.rmtree(str(f))
                except Exception:
                    pass


# 全局单例
cache_manager = CacheManager()