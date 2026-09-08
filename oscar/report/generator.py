"""Report Generator - produce audit report in Markdown and JSON formats.

Generates rich reports with evidence tables, line-level granularity,
categorized findings, and a category-weighted completeness score.

计分口径(category_policy.py 集中定义):
- 只有核心实现类别参与完整度计分,并按类别加权;外围类别(api 文档/
  license/checkpoint/外部资源/release)单独列出、不扣分。
- 计分分母只含 VERIFIED/INCOMPLETE/MISSING;UNCERTAIN / NOT_APPLICABLE /
  RESTRICTED / PLANNED 单列说明、不进计分域。
- 每个 finding 只渲染一次(旧实现因 findings 通道重复累积,标题出现 4 次)。
"""

import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, Optional

from oscar.models.schemas import (
    AuditFinding, AuditReport, AuditState, ClaimStatus, Evidence,
    EvidenceDetail, EvidenceType,
)
from oscar.report.category_policy import (
    CATEGORY_WEIGHTS, INFO_DISPLAY_ORDER, INFO_ONLY_CATEGORIES,
    SCORED_DISPLAY_ORDER,
)

# Human-readable titles for known claim IDs
CATEGORY_TITLES: Dict[str, str] = {
    "TRAINING-OVERALL": "Training Code Completeness",
    "INFERENCE-OVERALL": "Inference Functionality",
    "DEMO-OVERALL": "Demo Functionality",
    "EVALUATION-OVERALL": "Evaluation Functionality",
    "BENCHMARK-OVERALL": "Benchmark Functionality",
    "API-OVERALL": "API / Interface Completeness",
    "RES-DATASET": "Dataset Documentation",
    "RES-CHECKPOINT": "Checkpoint / Pretrained Model Documentation",
    "RES-EXTERNAL": "External Resource Documentation",
    "LIC-FILE": "Repository License File",
    "LIC-README": "README License Mention",
    "LIC-DEPS": "Dependency License Documentation",
    "LIC-HEADERS": "Source Code License Headers",
}

# Category → display name (used as group headings)
CATEGORY_DISPLAY: Dict[str, str] = {
    "core_method": "Core Methods",
    "training": "Training",
    "inference": "Inference",
    "evaluation": "Evaluation / Metrics",
    "dataset": "Dataset",
    "benchmark": "Benchmark",
    "demo": "Demo / Examples",
    "api_interface": "API / Interface",
    "license": "License",
    "checkpoint": "Checkpoints / Pretrained Weights",
    "external_resource": "External Resources",
    "release": "Release / Distribution",
}

_DOMAIN_STATUSES = {ClaimStatus.VERIFIED, ClaimStatus.INCOMPLETE, ClaimStatus.MISSING}
_SCORED_SET = set(CATEGORY_WEIGHTS.keys())


def _get_readable_title(claim_id: str, statement: str) -> str:
    """Get a human-readable title for the claim.

    Known IDs (TRAINING-OVERALL, API-OVERALL, etc.) are mapped to a fixed
    human-readable title.  All other claims (METHOD-*, LLM-*, etc.) use the
    *statement* directly, which should be the actual method/framework/module
    name from the paper — never the internal claim_id.
    """
    if claim_id in CATEGORY_TITLES:
        return CATEGORY_TITLES[claim_id]
    return statement


def _split_statement_title(statement: str) -> tuple[str, str]:
    """Split a ``"Name: functional description …"`` claim into a short title
    and a body paragraph.

    LLM 功能句 claim 是「名字: 一句话做什么」。整句当小节标题太长(用户
    反馈:标题应只保留冒号前的简短名字),冒号后的功能描述作为标题下的一
    段纯文本。无冒号的整句类 statement(如 benchmark 贡献句,无
    "名: 描述" 结构)若超过 96 字符,再按最靠近 96 的安全断点(", " / "; ")
    切成标题+正文;固定标题与中等长度语句
    原样返回,body 为空。
    """
    idx = statement.find(": ")
    if 0 < idx <= 70:
        head = statement[:idx].strip()
        tail = statement[idx + 2:].strip()
        if (
            head and tail and len(tail) >= 20
            and ":" not in head and ". " not in head and "  " not in head
        ):
            return head, tail
    if len(statement) > 96:
        for cut in range(96, 39, -1):
            if statement[cut] in ",;":
                head = statement[:cut].strip()
                tail = statement[cut + 1:].strip()
                if head and tail:
                    return head, tail
    return statement, ""


def _category_order(category) -> int:
    """Position of a category in its display list (scored first, info after)."""
    if category in SCORED_DISPLAY_ORDER:
        return SCORED_DISPLAY_ORDER.index(category)
    if category in INFO_DISPLAY_ORDER:
        return len(SCORED_DISPLAY_ORDER) + INFO_DISPLAY_ORDER.index(category)
    return len(SCORED_DISPLAY_ORDER) + len(INFO_DISPLAY_ORDER)


def _sort_key(finding: AuditFinding) -> tuple:
    return (_category_order(finding.category), finding.claim_id, finding.statement)


def generate_report(state: AuditState) -> AuditReport:
    """Generate the final audit report from audit state."""
    project_name = state.project.get("name", "Unknown")
    repo_url = state.project.get("repository_url", "")
    findings = sorted(state.findings.values(), key=_sort_key)
    claim_map = {c.claim_id: c for c in state.claims}

    # -- Evidence index ---------------------------------------------------
    # Collect once per unique finding (deduped by evidence_id). The old
    # findings×claims double loop inflated the index 4x on duplicate findings.
    all_evidence: list[Evidence] = []
    seen_ids: set[str] = set()
    for ev in state.evidences:
        if ev.evidence_id not in seen_ids:
            seen_ids.add(ev.evidence_id)
            all_evidence.append(ev)

    def add_evidence(ev: Evidence):
        if ev.evidence_id not in seen_ids:
            seen_ids.add(ev.evidence_id)
            all_evidence.append(ev)

    for finding in findings:
        claim = claim_map.get(finding.claim_id)
        if claim:
            for ev in claim.repository_evidence:
                add_evidence(ev)
            for ev in claim.external_evidence:
                add_evidence(ev)

    # Granular detail rows as evidence entries (deterministic numbering)
    for finding in findings:
        if finding.evidence_details:
            for detail in finding.evidence_details:
                if not detail.file_path and not detail.snippet and not detail.label:
                    continue
                counter = len(all_evidence) + 1
                add_evidence(Evidence(
                    evidence_id=f"E-DETAIL-{finding.claim_id}-{counter:03d}",
                    type=EvidenceType.REPOSITORY_FILE,
                    source="repository",
                    location=f"{detail.file_path}:{detail.line_number}",
                    content=detail.code_explanation or detail.snippet or detail.label or "",
                    supports=[finding.claim_id],
                ))

    # -- Stats & scoring --------------------------------------------------
    stats = {}
    for status in ClaimStatus:
        count = sum(1 for f in findings if f.status == status)
        if count > 0:
            stats[status.value] = count

    overall_score, category_stats = _compute_category_stats(findings)
    exec_summary = _generate_executive_summary(project_name, findings, stats, overall_score, category_stats, state)

    report = AuditReport(
        project_name=project_name,
        repository_url=repo_url,
        paper_title=state.paper.title if state.paper else None,
        paper_provided=state.paper is not None,
        executive_summary=exec_summary,
        findings=[f for f in findings if f.status not in
                  (ClaimStatus.UNCERTAIN, ClaimStatus.NOT_APPLICABLE)],
        uncertain_findings=[f for f in findings if f.status in
                            (ClaimStatus.UNCERTAIN, ClaimStatus.NOT_APPLICABLE)],
        evidence_index=all_evidence,
        summary_stats=stats,
        overall_score=overall_score if overall_score is not None else 0.0,
        category_stats=category_stats,
    )
    return report


def _compute_category_stats(findings: list[AuditFinding]) -> tuple[Optional[float], dict]:
    """Weighted per-category completeness score.

    score = Σ_c w_c·(v_c + 0.5·i_c)/n_c / Σ_c w_c  (n_c: category findings
    with VERIFIED/INCOMPLETE/MISSING status). Categories without domain
    findings are excluded entirely — absence of claims must not hurt.
    """
    by_category: dict = defaultdict(list)
    for f in findings:
        by_category[f.category].append(f)

    category_stats: dict[str, dict] = {}
    numerator = 0.0
    denominator = 0.0
    all_scored_found = False

    for cat, weight in CATEGORY_WEIGHTS.items():
        fs = by_category.get(cat, [])
        domain = [f for f in fs if f.status in _DOMAIN_STATUSES]
        verified = sum(1 for f in domain if f.status == ClaimStatus.VERIFIED)
        incomplete = sum(1 for f in domain if f.status == ClaimStatus.INCOMPLETE)
        entry = {
            "weight": weight,
            "scored": True,
            "count": len(domain),
            "verified": verified,
            "incomplete": incomplete,
            "missing": len(domain) - verified - incomplete,
            "uncertain": sum(1 for f in fs if f.status == ClaimStatus.UNCERTAIN),
            "not_applicable": sum(1 for f in fs if f.status == ClaimStatus.NOT_APPLICABLE),
            "restricted": sum(1 for f in fs if f.status == ClaimStatus.RESTRICTED),
            "planned": sum(1 for f in fs if f.status == ClaimStatus.PLANNED),
        }
        if domain:
            entry["score"] = round((verified + 0.5 * incomplete) / len(domain) * 100, 1)
            numerator += weight * entry["score"]
            denominator += weight
            all_scored_found = True
        else:
            entry["score"] = None
        category_stats[cat.value] = entry

    for cat in SCORED_DISPLAY_ORDER:
        if cat.value not in category_stats and cat not in INFO_ONLY_CATEGORIES:
            entry = {
                "weight": CATEGORY_WEIGHTS.get(cat, 1.0), "scored": True,
                "count": 0, "verified": 0, "incomplete": 0, "missing": 0,
                "uncertain": 0, "not_applicable": 0, "restricted": 0, "planned": 0,
                "score": None,
            }
            category_stats[cat.value] = entry

    # Informational categories: stats only, never scored
    # (INFO_ONLY_CATEGORIES 是 set,迭代必须先排序——set 迭代序跨进程随机,
    # 会破坏 JSON 键序的字节级确定性)
    for cat in sorted(INFO_ONLY_CATEGORIES, key=lambda c: c.value):
        fs = by_category.get(cat, [])
        category_stats[cat.value] = {
            "weight": 0.0,
            "scored": False,
            "count": len(fs),
            "verified": sum(1 for f in fs if f.status == ClaimStatus.VERIFIED),
            "incomplete": sum(1 for f in fs if f.status == ClaimStatus.INCOMPLETE),
            "missing": sum(1 for f in fs if f.status == ClaimStatus.MISSING),
            "uncertain": sum(1 for f in fs if f.status == ClaimStatus.UNCERTAIN),
            "not_applicable": sum(1 for f in fs if f.status == ClaimStatus.NOT_APPLICABLE),
            "restricted": sum(1 for f in fs if f.status == ClaimStatus.RESTRICTED),
            "planned": sum(1 for f in fs if f.status == ClaimStatus.PLANNED),
            "score": None,
        }

    overall = numerator / denominator if denominator > 0 else (None if not all_scored_found else 0.0)
    return overall, category_stats


def _generate_executive_summary(
    project_name: str,
    findings: list[AuditFinding],
    stats: dict,
    overall_score: Optional[float],
    category_stats: dict,
    state: AuditState,
) -> str:
    """Generate an executive summary with the weighted score and a category table."""
    verified = stats.get(ClaimStatus.VERIFIED.value, 0)
    incomplete = stats.get(ClaimStatus.INCOMPLETE.value, 0)
    missing = stats.get(ClaimStatus.MISSING.value, 0)
    restricted = stats.get(ClaimStatus.RESTRICTED.value, 0)
    uncertain = stats.get(ClaimStatus.UNCERTAIN.value, 0)
    na = stats.get(ClaimStatus.NOT_APPLICABLE.value, 0)

    lines = [
        "## Executive Summary",
        "",
        f"**Project:** {project_name}",
        "",
    ]

    # Project-level overview sentence produced by repository analyzer (if any)
    manifest = state.repository_manifest
    if manifest and manifest.code_summary and manifest.code_summary.project_summary:
        lines.append(f"*{manifest.code_summary.project_summary}*")
        lines.append("")

    # 无论文运行:claims 来自仓库自己的 README —— 明确标注,不冒充论文核对
    if state.paper is None:
        lines.append("*No paper was provided — claims are extracted from the "
                     "repository's own README, so this run checks the repo's "
                     "self-description, not paper fidelity.*")
        lines.append("")

    if overall_score is not None:
        lines.append(f"**Overall Completeness Score:** {overall_score:.1f}%")
        lines.append("")
        lines.append("**Weighted across core implementation categories** "
                     "(core methods, training, inference, evaluation, dataset, "
                     "benchmark, demo). Informational items "
                     "(API docs, licenses, external resources, checkpoints, "
                     "releases) are listed separately and do not affect the score.")
    else:
        lines.append("**Overall Completeness Score:** not computed — no "
                     "scoreable category had findings.")
        lines.append("")

    # Category score table
    header_row = ["Category", "Weight", "Verified", "Incomplete", "Missing", "Score"]
    lines.append("| Category | Weight | Verified | Incomplete | Missing | Score |")
    lines.append("|---|---|---|---|---|---|")
    for cat_key in [c.value for c in SCORED_DISPLAY_ORDER]:
        entry = category_stats.get(cat_key)
        if not entry or not entry.get("count"):
            continue
        score_str = f"{entry['score']:.1f}%" if entry.get("score") is not None else "-"
        weight_str = f"{entry['weight']:g}"
        lines.append(
            f"| {CATEGORY_DISPLAY.get(cat_key, cat_key)} | {weight_str} | "
            f"{entry['verified']} | {entry['incomplete']} | {entry['missing']} | {score_str} |"
        )
    lines.append("")

    # Status totals & interpretation
    lines.append(f"**Summary Statistics:**  Verified {verified} · Incomplete "
                 f"{incomplete} · Missing {missing} · Restricted {restricted} · "
                 f"Uncertain {uncertain} · Not applicable {na} · "
                 f"Total {len(findings)}")
    lines.append("")

    if overall_score is not None:
        if overall_score >= 80:
            interp = ("The core implementation of the project appears complete: "
                      "most claimed components are backed by concrete code.")
        elif overall_score >= 50:
            interp = ("The project has moderate completeness: several core "
                      "components are incomplete or missing. Review the detailed "
                      "findings for specifics.")
        else:
            interp = ("The project has low completeness: significant core "
                      "components are missing or not backed by code.")
        lines.append(f"**Interpretation:** {interp}")
        lines.append("")

    notes = []
    if restricted > 0:
        notes.append(f"{restricted} component(s) are explicitly restricted "
                     "(copyright / proprietary / licensing constraints).")
    if uncertain or na:
        notes.append(f"{uncertain + na} finding(s) could not be determined "
                     "(uncertain or not applicable) and are excluded from the score.")
    if notes:
        lines.append("**Notes:** " + " ".join(notes))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _render_evidence_details_table(details: list[EvidenceDetail]) -> str:
    """Render evidence details as a Markdown table (incl. what-the-code-does)."""
    if not details:
        return "_No granular evidence available._"

    lines = [
        "| File | Line | Class/Method | Detail | Snippet | What the code does |",
        "|------|------|-------------|--------|---------|--------------------|",
    ]
    for d in details[:15]:  # Limit to 15 rows
        line = str(d.line_number) if d.line_number else "-"
        cls = d.class_name or ""
        func = d.function_name or ""
        cls_func = (f"{cls}.{func}" if cls and func else cls or func or "-")
        snippet = d.snippet[:60].replace("|", "\\|").replace("\n", " ")
        label = d.label[:40].replace("|", "\\|")
        expl = d.code_explanation[:140].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {d.file_path} | {line} | {cls_func} | {label} | `{snippet}` | {expl} |")
    if len(details) > 15:
        lines.append(f"| ... | ... | ... | ... | ... | *{len(details) - 15} more entries* |")
    return "\n".join(lines)


def _finding_block(finding: AuditFinding, heading_level: str) -> list[str]:
    """Render one finding as a block of markdown lines."""
    title, body = _split_statement_title(
        _get_readable_title(finding.claim_id, finding.statement)
    )
    lines = [f"{heading_level} {title}", ""]
    if body:
        lines.append(body)
        lines.append("")
    lines.append(f"- **Claim ID:** `{finding.claim_id}`")
    lines.append(f"- **Status:** `{finding.status.value}`")
    lines.append(f"- **Confidence:** `{finding.confidence:.0%}`")
    lines.append(f"- **Evidence Summary:** {finding.evidence_summary}")
    lines.append("")
    lines.append("#### Evidence Details")
    lines.append("")
    lines.append(_render_evidence_details_table(finding.evidence_details))
    lines.append("")
    lines.append("#### Explanation")
    lines.append("")
    explanation = finding.explanation
    if "Evidence Table:" in explanation:
        explanation = explanation.split("Evidence Table:")[0].strip()
    lines.append(explanation)
    lines.append("")
    return lines


def save_markdown_report(report: AuditReport, output_path: str):
    """Save the report as Markdown with rich evidence tables.

    Layout: every finding renders exactly once, grouped by category in
    importance order (scored categories first, informational next, then
    excluded statuses). No duplicate titles.
    """
    lines = []
    lines.append("# Open-Source Completeness Audit Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(f"**Project:** {report.project_name}")
    lines.append(f"**Repository:** {report.repository_url}")
    if report.paper_title:
        lines.append(f"**Paper:** {report.paper_title}")
    elif not report.paper_provided:
        lines.append("**Paper:** not provided — claims are extracted from the "
                     "repository's own README (self-description check)")
    lines.append("")

    lines.append(report.executive_summary)
    lines.append("")

    all_findings = sorted(report.findings + report.uncertain_findings, key=_sort_key)

    # Partition: scored / informational / excluded (uncertain, NA, restricted, planned)
    scored = [f for f in all_findings if f.category in _SCORED_SET and f.status in _DOMAIN_STATUSES]
    excluded = [f for f in all_findings if f.category in _SCORED_SET and f.status not in _DOMAIN_STATUSES]
    info = [f for f in all_findings if f.category not in _SCORED_SET]

    def emit_category_group(category, items, note: str):
        weight = CATEGORY_WEIGHTS.get(category)
        title = CATEGORY_DISPLAY.get(category.value, category.value)
        head = f"### {title}"
        if weight is not None:
            head += f" — weight {weight:g}"
        if note:
            head += f"  *({note})*"
        lines.append(head)
        lines.append("")
        for finding in items:
            lines.extend(_finding_block(finding, "####"))
        lines.append("")

    # Scored findings, grouped by category in importance order
    lines.append("## Scored Findings")
    lines.append("")
    lines.append("These findings determine the weighted completeness score.")
    lines.append("")
    emitted_any = False
    for category in SCORED_DISPLAY_ORDER:
        items = [f for f in scored if f.category == category]
        if items:
            emit_category_group(category, items, "")
            emitted_any = True
    if not emitted_any:
        lines.append("_No scoreable findings in this run._")
        lines.append("")

    # Informational findings (documented separately; do not affect the score)
    lines.append("## Informational (Not Scored)")
    lines.append("")
    lines.append("API/interface, license, checkpoint, external-resource and "
                 "release items are listed for reference; their absence does "
                 "not reduce the completeness score.")
    lines.append("")
    info_emitted = False
    for category in INFO_DISPLAY_ORDER:
        items = [f for f in info if f.category == category]
        if items:
            emit_category_group(category, items, "")
            info_emitted = True
    if not info_emitted:
        lines.append("_None._")
        lines.append("")

    # Excluded statuses on scored categories (uncertain / NA / restricted / planned)
    if excluded:
        lines.append("## Excluded from Scoring")
        lines.append("")
        lines.append("Findings below carry a status that does not contribute to "
                     "the weighted score (uncertain, not applicable, restricted, "
                     "or planned).")
        lines.append("")
        for category in SCORED_DISPLAY_ORDER:
            items = sorted([f for f in excluded if f.category == category], key=lambda f: f.claim_id)
            if items:
                emit_category_group(category, items, "not counted")
        lines.append("")

    # Evidence Index
    if report.evidence_index:
        lines.append("## Evidence Index")
        lines.append("")
        lines.append("| ID | Type | Location | Supports |")
        lines.append("|----|------|----------|----------|")
        for ev in report.evidence_index[:60]:
            loc = ev.location[:60] if ev.location else "-"
            supports = ", ".join(ev.supports[:3]) if ev.supports else "-"
            lines.append(f"| {ev.evidence_id} | {ev.type.value} | {loc} | {supports} |")
        lines.append("")

    # Closing Summary
    lines.append("---")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total Findings | {len(all_findings)} |")
    for status in sorted(report.summary_stats):
        lines.append(f"| {status} | {report.summary_stats[status]} |")
    lines.append(f"| Evidence Entries | {len(report.evidence_index)} |")
    if report.overall_score is not None and (report.findings or report.summary_stats):
        lines.append(f"| Overall Completeness Score | {report.overall_score:.1f}% |")
    lines.append("")
    lines.append("*Report generated by OSCAR - Open-Source Completeness Audit & Review*")
    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_json_report(report: AuditReport, output_path: str):
    """Save the report as JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2, ensure_ascii=False, default=str)
