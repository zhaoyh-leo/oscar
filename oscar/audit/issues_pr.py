"""GitHub Issues / PR Investigation.

Investigates GitHub Issues and PRs for MISSING, INCOMPLETE, and UNCERTAIN findings.
"""

import re
from typing import Optional

from oscar.llm.client import llm_client
from oscar.prompts import build_issue_analysis_prompt
from oscar.models.schemas import (
    AuditFinding, AuditState, ClaimStatus, Evidence, EvidenceType, IssueResult,
)
from oscar.config import config
from oscar.utils.github import search_issues, fetch_issue_comments, parse_github_url


def investigate_issues(state: AuditState) -> list[IssueResult]:
    """Investigate GitHub Issues/PRs for unresolved findings.

    Only triggered for MISSING, INCOMPLETE, and UNCERTAIN findings.
    """
    repo_url = state.project.get("repository_url", "")
    if not repo_url:
        return []

    # Find which findings need investigation
    unresolved = [f for f in state.findings.values() if f.status in (
        ClaimStatus.MISSING, ClaimStatus.INCOMPLETE, ClaimStatus.UNCERTAIN
    )]

    if not unresolved:
        return []

    # Search for relevant issues
    keywords = config.audit.issue_search_keywords
    issues = search_issues(repo_url, keywords, max_results=config.audit.max_issues_to_fetch)

    results = []
    for issue in issues[:10]:  # Limit to 10 most relevant
        # Fetch comments
        comments = fetch_issue_comments(repo_url, issue["number"])

        # Determine which claims this issue relates to
        relevant_claims = []
        body_text = (issue.get("title", "") + " " + (issue.get("body", "") or "")).lower()
        for comment in comments:
            body_text += " " + (comment.get("body", "") or "").lower()

        for finding in unresolved:
            # 词级宽松匹配曾把所有含 'a/the/of' 等停用词的 finding 判为与每个
            # issue 相关,导致无关 issue(如仅询问工具脚本用法)把实现完
            # 整的方法 claim 改写为 RESTRICTED。改为:≥2 个实义词同时命中才相关。
            tokens = _substantive_tokens(finding.statement)
            if sum(1 for kw in tokens if kw in body_text) >= 2:
                relevant_claims.append(finding.claim_id)

        # Extract key statements from issue and comments
        key_statements = []
        if issue.get("body"):
            key_statements.append(f"Issue body: {issue['body'][:200]}")
        for comment in comments[:3]:
            if comment.get("body"):
                key_statements.append(f"Comment by {comment['author']}: {comment['body'][:200]}")

        # Use LLM to analyze if issue explains the missing component
        explanation = None
        implication = None
        if relevant_claims:
            analysis = _analyze_issue_llm(issue, comments, relevant_claims)
            if analysis:
                explanation = analysis[0]
                implication = analysis[1]

        issue_result = IssueResult(
            issue_number=issue["number"],
            title=issue["title"],
            body=issue.get("body", "")[:500] if issue.get("body") else None,
            state=issue["state"],
            author=issue.get("author"),
            is_pr=issue.get("is_pr", False),
            relevant_claims=relevant_claims,
            key_statements=key_statements[:5],
            explanation=explanation,
            status_implication=implication,
        )
        results.append(issue_result)

    return results


_STOPWORDS = {
    "a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "is", "are",
    "was", "be", "it", "this", "that", "with", "as", "by", "we", "our", "can",
    "not", "no", "at", "from", "its", "their", "does", "do", "have", "has",
}


def _substantive_tokens(text: str) -> list[str]:
    """Lowercased alpha tokens minus stopwords (deterministic order)."""
    return sorted(
        {
            t for t in re.findall(r"[a-z][a-z0-9']*", (text or "").lower())
            if t not in _STOPWORDS
        }
    )


def _analyze_issue_llm(
    issue: dict, comments: list[dict], relevant_claims: list[str]
) -> Optional[tuple[str, str]]:
    """Use LLM to analyze if an issue explains the missing component.

    Returns ``(explanation, status_implication)``. The implication is the
    LLM's STRUCTURED field (PLANNED / RESTRICTED / MISSING / UNCERTAIN) —
    downstream must never substring-scan the explanation prose, which
    misreads negations ("no copyright restriction" ⇒ RESTRICTED).
    """
    prompt = build_issue_analysis_prompt(issue, comments, relevant_claims)

    try:
        result = llm_client.chat_json([{"role": "user", "content": prompt}], temperature=0.1)
        if isinstance(result, dict) and result.get("relevant", False):
            implication = str(result.get("status_implication", "UNCERTAIN")).upper()
            if implication not in ("PLANNED", "RESTRICTED", "MISSING", "UNCERTAIN"):
                implication = "UNCERTAIN"
            return (
                str(result.get("explanation", "Relevant issue found."))[:500],
                implication,
            )
    except Exception:
        pass
    return None