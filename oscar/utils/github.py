"""GitHub API helpers."""

import re
from typing import Optional

import httpx


def parse_github_url(url: str) -> dict:
    """Parse a GitHub URL to extract owner, repo, and other info."""
    patterns = [
        r"https?://github\.com/([^/]+)/([^/]+)",
        r"git@github\.com:([^/]+)/([^/]+)\.git",
    ]
    for pattern in patterns:
        m = re.match(pattern, url)
        if m:
            owner, repo = m.group(1), m.group(2).replace(".git", "")
            return {"owner": owner, "repo": repo, "url": f"https://github.com/{owner}/{repo}"}
    return {"owner": "", "repo": "", "url": url}


def check_repo_exists(url: str) -> Optional[bool]:
    """Check if a GitHub repository exists.

    Returns:
      - True:  仓库确定存在（HTTP 200）
      - False: 仓库确定不存在（HTTP 404）
      - None:  网络错误/限流，无法确定 —— 调用方应视为“可能存在”继续尝试，
               而不是据此放弃（连接不稳定时 API 超时不应误判为仓库不存在）
    """
    parsed = parse_github_url(url)
    if not parsed["owner"]:
        return False
    api_url = f"https://api.github.com/repos/{parsed['owner']}/{parsed['repo']}"
    try:
        resp = httpx.get(api_url, timeout=10)
        if resp.status_code == 200:
            return True
        if resp.status_code == 404:
            return False
        return None  # 限流或其他错误，无法确定
    except Exception:
        return None  # 网络错误，无法确定


def fetch_repo_metadata(url: str) -> Optional[dict]:
    """Fetch repository metadata from GitHub API."""
    parsed = parse_github_url(url)
    if not parsed["owner"]:
        return None
    api_url = f"https://api.github.com/repos/{parsed['owner']}/{parsed['repo']}"
    try:
        resp = httpx.get(api_url, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def search_issues(url: str, keywords: list[str], max_results: int = 50) -> list[dict]:
    """Search for issues containing specific keywords."""
    parsed = parse_github_url(url)
    if not parsed["owner"]:
        return []

    results = []

    # Use simpler search approach
    api_url = f"https://api.github.com/search/issues?q=repo:{parsed['owner']}/{parsed['repo']}+type:issue&per_page={min(max_results, 100)}&sort=updated"
    try:
        resp = httpx.get(api_url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            for item in data.get("items", []):
                # Check if any keyword matches
                body = (item.get("title", "") + " " + (item.get("body", "") or "")).lower()
                if any(kw.lower() in body for kw in keywords):
                    results.append({
                        "number": item["number"],
                        "title": item["title"],
                        "body": item.get("body", ""),
                        "state": item["state"],
                        "author": item["user"]["login"] if item.get("user") else None,
                        "is_pr": "pull_request" in item,
                        "url": item["html_url"],
                    })
            return results
    except Exception:
        pass
    return results


def fetch_issue_comments(url: str, issue_number: int) -> list[dict]:
    """Fetch comments for a specific issue."""
    parsed = parse_github_url(url)
    if not parsed["owner"]:
        return []
    api_url = f"https://api.github.com/repos/{parsed['owner']}/{parsed['repo']}/issues/{issue_number}/comments"
    try:
        resp = httpx.get(api_url, timeout=10)
        if resp.status_code == 200:
            return [
                {
                    "author": c["user"]["login"] if c.get("user") else None,
                    "body": c.get("body", ""),
                }
                for c in resp.json()
            ]
    except Exception:
        pass
    return []