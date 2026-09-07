"""Evidence Aggregator - fuse evidence from multiple sources.

Evidence priority:
  1. Author/maintainer official statements
  2. Repository direct code evidence
  3. README / Paper
  4. Maintainer / Issue replies
  5. Regular issue user statements
  6. LLM inference
"""

from typing import Optional

from oscar.models.schemas import (
    AuditFinding, AuditState, Claim, ClaimStatus, Evidence, EvidenceType, IssueResult,
)


def aggregate_evidence(
    state: AuditState,
    findings: dict[str, AuditFinding],
    issue_results: list[IssueResult],
) -> dict[str, AuditFinding]:
    """Aggregate evidence from all sources.

    Returns only the findings that were updated, keyed by claim_id — the
    findings channel reducer merges them over the originals (后写覆盖).
    All write-backs deep-copy first: the objects in ``state.findings`` must
    never be mutated in place (parallel audit nodes may hold the same
    references in earlier supersteps).

    状态裁决权说明:规则层只处理「官方陈述」级证据(issue 中的
    RESTRICTED/PLANNED/MISSING);planner 的弱映射证据不再直接把 UNCERTAIN
    抬成 VERIFIED —— 那是 keyword/名字级匹配,真正的代码裁决交给
    evidence_grounder(LLM 读代码原文),规则层只补充 evidence_summary。
    """
    working: dict[str, AuditFinding] = dict(findings)
    updated: dict[str, AuditFinding] = {}

    def set_finding(claim_id: str, finding: AuditFinding) -> None:
        working[claim_id] = finding
        updated[claim_id] = finding

    # Process issue results to update findings
    for issue in issue_results:
        for claim_id in issue.relevant_claims:
            finding = working.get(claim_id)
            if finding is None or not issue.explanation:
                continue
            implied_status = _extract_status_from_issue(issue)
            if implied_status and _is_higher_priority(implied_status, finding.status):
                new = finding.model_copy(deep=True)
                new.status = implied_status
                new.confidence = min(new.confidence + 0.2, 1.0)
                new.explanation = (
                    new.explanation
                    + f"\n\nIssue #{issue.issue_number}: {issue.explanation}"
                ).strip()
                set_finding(claim_id, new)

    # Reconcile findings with original claims (evidence summary enrichment only)
    for claim in state.claims:
        finding = working.get(claim.claim_id)
        if finding is None:
            continue
        new = _enrich_from_claim_evidence(finding, claim)
        if new is not None:
            set_finding(claim.claim_id, new)

    return updated


def _extract_status_from_issue(issue: IssueResult) -> Optional[ClaimStatus]:
    """Extract implied status from the issue investigation.

    只信 LLM 的结构化 status_implication 字段——绝不对 explanation 散文做
    子串扫描:『no copyright restriction / not restricted』这类否定句会命中
    RESTRICTED/COPYRIGHT 把状态误抬为 RESTRICTED(历史上有无关 issue 因此
    把已实现 claim 误标 RESTRICTED)。
    """
    if not issue.explanation:
        return None
    implication = (issue.status_implication or "").upper()
    mapping = {
        "RESTRICTED": ClaimStatus.RESTRICTED,
        "PLANNED": ClaimStatus.PLANNED,
        "MISSING": ClaimStatus.MISSING,
    }
    return mapping.get(implication)


def _is_higher_priority(new: ClaimStatus, old: ClaimStatus) -> bool:
    """Check if the new status is higher priority (more informative) than old."""
    priority = {
        ClaimStatus.RESTRICTED: 1,
        ClaimStatus.PLANNED: 2,
        ClaimStatus.VERIFIED: 3,
        ClaimStatus.INCOMPLETE: 4,
        ClaimStatus.MISSING: 5,
        ClaimStatus.UNCERTAIN: 6,
        ClaimStatus.NOT_APPLICABLE: 7,
    }
    return priority.get(new, 99) < priority.get(old, 99)


def _enrich_from_claim_evidence(finding: AuditFinding, claim: Claim) -> Optional[AuditFinding]:
    """Enrich evidence_summary with claim-level repository evidence locations.

    Returns a deep-copied finding when changed, else None. Never changes the
    status — verdicts based on repository code are produced by
    evidence_grounder (LLM reads the actual code), not by keyword/name-level
    mapping evidence.
    """
    repo_evidence = [ev for ev in claim.repository_evidence if ev.type == EvidenceType.REPOSITORY_FILE]
    if not repo_evidence or finding.evidence_summary:
        return None

    locations: list[str] = []
    for ev in repo_evidence:
        for loc in (ev.location or "").split(","):
            loc = loc.strip()
            if loc and loc not in locations:
                locations.append(loc)
        if len(locations) >= 5:
            break

    if not locations:
        return None

    new = finding.model_copy(deep=True)
    new.evidence_summary = "Files: " + ", ".join(locations)
    return new
