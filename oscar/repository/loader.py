"""Repository loader - clone and prepare repositories for analysis."""

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

from oscar.config import config
from oscar.utils.github import parse_github_url, check_repo_exists
from oscar.utils.cache import REPO_CACHE_TTL, repo_cache_get, repo_cache_set
from oscar.utils.file_utils import rmtree_robust

logger = logging.getLogger(__name__)


def clone_repository(url: str, target_dir: Optional[str] = None) -> str:
    """Clone a GitHub repository for analysis; returns a valid local path.

    返回的路径保证存在且含 .git —— 它可能是:
    - 磁盘缓存的克隆(.oscar_cache/repos/<key>,提升成功时);
    - 原位克隆(clones/<repo>,提升失败时——本次直接使用,不阻塞审计)。

    为什么没有 clones/<repo> → 缓存的 symlink:
    Windows 默认无目录 symlink 权限(实测 WinError 5),旧实现的 symlink 在
    Windows 上必然失败,故改为直接返回缓存目录路径,调用方只读使用,无需链接。

    连接鲁棒性(针对 GitHub 连接不稳定场景):
    - 缓存检查在最前:命中缓存时零网络调用。否则 check_repo_exists 的 API
      请求在弱网下会白白等待 timeout 秒,缓存已存在时完全没有必要。
    - check_repo_exists 返回 None(网络错误/限流)时不放弃,继续尝试克隆;
      只有确定 404 才报"仓库不存在"。
    - git clone 按 max_clone_retries 重试,单次超时由 clone_timeout 控制
      (该值是"单次 git clone 的等待上限",git 快速失败时不会等满),
      重试间隔逐步增大,避免在弱网下直接失败。
    """
    parsed = parse_github_url(url)
    repo_name = parsed["repo"] or "unknown_repo"

    if target_dir is None:
        target_dir = str(config.paths.clone_dir / repo_name)

    # 1) 缓存优先:命中(缓存目录即真实仓库)直接复用,零网络调用
    cached = repo_cache_get(url)
    if cached:
        return cached

    # 2) 原位克隆仍在且新鲜(缓存被手动清理过,如用户删了 .oscar_cache)
    #    → 直接提升复用,同样零网络调用;过期则删掉重克隆。
    if os.path.isdir(os.path.join(target_dir, ".git")):
        if time.time() - os.path.getmtime(target_dir) < REPO_CACHE_TTL:
            logger.info("Reusing in-place clone at %s", target_dir)
            promoted = repo_cache_set(url, target_dir)
            return promoted or target_dir
        logger.info("In-place clone at %s is stale; removing.", target_dir)
        if not rmtree_robust(target_dir):
            raise RuntimeError(
                f"Cannot remove stale clone at {target_dir}: files are locked "
                "(antivirus scanning a previous clone?). Delete it manually and retry."
            )

    if check_repo_exists(url) is False:
        raise ValueError(f"Repository does not exist or is not accessible: {url}")

    clone_url = url if url.endswith(".git") else f"{url}.git"

    last_error = "unknown error"
    for attempt in range(1, config.audit.max_clone_retries + 1):
        # 清理上次失败留下的半成品 clone(锁定文件带退避重试)。
        # 清理不掉就明确报错,而不是让 git 报 "destination not empty"。
        if os.path.exists(target_dir):
            if not rmtree_robust(target_dir):
                raise RuntimeError(
                    f"Cannot clean up partial clone at {target_dir}: files are locked "
                    "(likely antivirus scanning the previous clone). "
                    "Retry in a moment or delete the directory manually."
                )

        try:
            result = subprocess.run(
                ["git", "clone", "--depth", "1", clone_url, target_dir],
                capture_output=True,
                text=True,
                timeout=config.audit.clone_timeout,
            )
            if result.returncode == 0:
                # 克隆成功 → 尽力提升入缓存。提升本身绝不致命:
                # 失败时克隆保留在 target_dir,本次运行照常使用(旧实现把提升失败
                # 误判为克隆失败,导致重试把好克隆删掉,最终 3 次全挂)。
                promoted = repo_cache_set(url, target_dir)
                if promoted:
                    logger.info("Cloned %s and promoted to cache: %s", repo_name, promoted)
                else:
                    logger.warning(
                        "Clone succeeded but cache promotion failed (files locked by "
                        "antivirus?); using in-place clone at %s",
                        target_dir,
                    )
                return promoted or target_dir
            last_error = result.stderr.strip() or f"git clone returned {result.returncode}"
        except subprocess.TimeoutExpired:
            last_error = f"git clone timed out after {config.audit.clone_timeout}s"
        except Exception as e:  # noqa: BLE001
            last_error = str(e)

        # 重试可见性:让调用方(CLI/日志)能观察到重试正在发生
        if attempt < config.audit.max_clone_retries:
            logger.warning(
                "git clone attempt %d/%d failed (%s); retrying in %ds...",
                attempt, config.audit.max_clone_retries, last_error, 2 * attempt,
            )
            time.sleep(2 * attempt)

    raise RuntimeError(
        f"Failed to clone repository after {config.audit.max_clone_retries} attempts: {last_error}"
    )


def get_readme_content(repo_path: str) -> Optional[str]:
    """Get README content from a cloned repository."""
    for name in ["README.md", "README.rst", "README.txt", "README", "Readme.md"]:
        path = os.path.join(repo_path, name)
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
            except Exception:
                pass
    return None


def get_repo_structure(repo_path: str, max_depth: int = 3) -> list[str]:
    """Get a simplified directory structure of the repository."""
    structure = []
    repo_path = Path(repo_path)
    for root, dirs, files in os.walk(repo_path):
        # Skip .git and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith((".", "__pycache__"))]
        level = Path(root).relative_to(repo_path).parts
        if len(level) > max_depth:
            continue
        indent = "  " * len(level)
        structure.append(f"{indent}{Path(root).name}/")
        for f in files[:20]:  # Limit files per dir
            structure.append(f"{indent}  {f}")
    return structure