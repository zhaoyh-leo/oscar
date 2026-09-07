"""Inference / Demo / Evaluation Completeness Audit."""

import os
from typing import Optional

from oscar.mapping.claim_code_mapper import map_claim_to_code
from oscar.models.schemas import (
    AuditFinding, AuditState, Claim, ClaimCategory, ClaimStatus,
    Evidence, EvidenceDetail, EvidenceType, MappingResult, RepositoryManifest,
)
from oscar.utils.file_utils import read_file


def audit_inference_eval(state: AuditState) -> list[AuditFinding]:
    """Audit inference, demo, evaluation, and benchmark completeness."""
    findings = []
    manifest = state.repository_manifest
    repo_path = state.project.get("clone_path", "")

    if not manifest:
        return findings

    # Check each category
    categories = {
        "INFERENCE": _check_inference(manifest, repo_path),
        "DEMO": _check_demo(manifest),
        "EVALUATION": _check_evaluation(manifest, repo_path),
    }

    for cat_name, (found, evidence, details, evidence_details) in categories.items():
        claim_id = f"{cat_name}-OVERALL"
        if found:
            status = ClaimStatus.VERIFIED
            confidence = 0.9
        elif evidence:  # Partial evidence
            status = ClaimStatus.INCOMPLETE
            confidence = 0.5
        else:
            status = ClaimStatus.MISSING
            confidence = 0.2

        finding = AuditFinding(
            claim_id=claim_id,
            category=_get_category(cat_name),
            statement=f"{cat_name.capitalize()} functionality",
            status=status,
            confidence=confidence,
            evidence_summary=details or "No evidence found",
            evidence_details=evidence_details,
            explanation=details or f"No {cat_name.lower()} related code found",
        )
        findings.append(finding)

    # Benchmark: 论文/README 明确声称 "introduce a benchmark" 时(BENCHMARK 类
    # claim),逐条审计这些 claim(走 mapper + grounding,
    # 与 core_method 同路径)——文件名关键词式的 OVERALL 检查只作无 claim 时的
    # 兜底。此前 BENCHMARK 类别没有任何模块消费 claims,论文的 benchmark 贡献
    # 被静默丢弃,报告只剩一个基于噪音检索的 MISSING OVERALL。
    bench_claims = [
        c for c in (state.claims or [])
        if c.category == ClaimCategory.BENCHMARK
    ]
    if bench_claims:
        for claim in bench_claims:
            findings.append(_audit_benchmark_claim(claim, manifest, repo_path))
    else:
        found, evidence, details, evidence_details = _check_benchmark(manifest)
        if found:
            status, confidence = ClaimStatus.VERIFIED, 0.9
        elif evidence:
            status, confidence = ClaimStatus.INCOMPLETE, 0.5
        else:
            status, confidence = ClaimStatus.MISSING, 0.2
        findings.append(AuditFinding(
            claim_id="BENCHMARK-OVERALL",
            category=ClaimCategory.BENCHMARK,
            statement="Benchmark functionality",
            status=status,
            confidence=confidence,
            evidence_summary=details or "No evidence found",
            evidence_details=evidence_details,
            explanation=details or "No benchmark related code found",
        ))

    return findings


def _audit_benchmark_claim(
    claim: Claim,
    manifest: RepositoryManifest,
    repo_path: str,
) -> AuditFinding:
    """Audit one benchmark-category paper claim (mirrors core_method).

    状态只是暂态(mapper 置信度),真正的裁决在 grounding 阶段:检索到的
    真实代码块会喂给 LLM 重判并回写 evidence 定位。
    """
    mapping = map_claim_to_code(claim, manifest, repo_path)

    if mapping.decision == "MATCH":
        status = ClaimStatus.VERIFIED if mapping.confidence >= 0.7 else ClaimStatus.INCOMPLETE
    elif mapping.confidence > 0:
        status = ClaimStatus.INCOMPLETE
    else:
        status = ClaimStatus.MISSING

    evidence_details = []
    for f in mapping.candidate_files[:12]:
        evidence_details.append(EvidenceDetail(
            file_path=f,
            label="candidate file",
        ))

    return AuditFinding(
        claim_id=claim.claim_id,
        category=ClaimCategory.BENCHMARK,
        statement=claim.statement,
        status=status,
        confidence=mapping.confidence,
        evidence_summary=(
            f"Files: {', '.join(mapping.candidate_files[:5])}"
            if mapping.candidate_files else "No matching files found"
        ),
        evidence_details=evidence_details,
        explanation=mapping.reason,
    )


def _get_category(name: str) -> ClaimCategory:
    mapping = {
        "INFERENCE": ClaimCategory.INFERENCE,
        "DEMO": ClaimCategory.DEMO,
        "EVALUATION": ClaimCategory.EVALUATION,
        "BENCHMARK": ClaimCategory.BENCHMARK,
    }
    return mapping.get(name, ClaimCategory.INFERENCE)


def _check_inference(manifest: RepositoryManifest, repo_path: str) -> tuple[bool, Optional[Evidence], str, list[EvidenceDetail]]:
    """Check for inference code."""
    keywords = ["infer", "predict", "test", "forward"]
    found_files = []
    evidence = None

    for f in manifest.files:
        f_lower = f.lower()
        if any(kw in f_lower for kw in keywords):
            found_files.append(f)

    evidence_details = []
    for f in found_files[:5]:
        evidence_details.append(EvidenceDetail(file_path=f, label="inference file"))

    if found_files:
        evidence = Evidence(
            evidence_id="E-INFER",
            type=EvidenceType.REPOSITORY_FILE,
            source="github",
            location=", ".join(found_files[:5]),
            content=f"Inference-related files found",
            supports=["INFERENCE-OVERALL"],
        )
        return True, evidence, f"Inference files: {', '.join(found_files[:5])}", evidence_details

    return False, None, "No inference-specific code found", evidence_details


def _check_demo(manifest: RepositoryManifest) -> tuple[bool, Optional[Evidence], str, list[EvidenceDetail]]:
    """Check for demo code."""
    keywords = ["demo", "example", "notebook", "app"]
    found_files = []

    for f in manifest.files:
        f_lower = f.lower()
        if any(kw in f_lower for kw in keywords):
            found_files.append(f)

    evidence_details = []
    for f in found_files[:5]:
        evidence_details.append(EvidenceDetail(file_path=f, label="demo file"))

    if found_files:
        return True, Evidence(
            evidence_id="E-DEMO",
            type=EvidenceType.REPOSITORY_FILE,
            source="github",
            location=", ".join(found_files[:5]),
            content=f"Demo-related files found",
            supports=["DEMO-OVERALL"],
        ), f"Demo files: {', '.join(found_files[:5])}", evidence_details

    return False, None, "No demo code found", evidence_details


def _check_evaluation(manifest: RepositoryManifest, repo_path: str) -> tuple[bool, Optional[Evidence], str, list[EvidenceDetail]]:
    """Check for evaluation code."""
    keywords = ["eval", "metric", "score", "accuracy", "evaluate"]
    found_files = []
    evidence = None

    for f in manifest.files:
        f_lower = f.lower()
        if any(kw in f_lower for kw in keywords):
            found_files.append(f)

    evidence_details = []
    for f in found_files[:5]:
        evidence_details.append(EvidenceDetail(file_path=f, label="evaluation file"))

    if found_files:
        return True, Evidence(
            evidence_id="E-EVAL",
            type=EvidenceType.REPOSITORY_FILE,
            source="github",
            location=", ".join(found_files[:5]),
            content=f"Evaluation-related files found",
            supports=["EVALUATION-OVERALL"],
        ), f"Evaluation files: {', '.join(found_files[:5])}", evidence_details

    return False, None, "No evaluation code found", evidence_details


def _check_benchmark(manifest: RepositoryManifest) -> tuple[bool, Optional[Evidence], str, list[EvidenceDetail]]:
    """Check for benchmark code."""
    keywords = ["benchmark", "bench"]
    found_files = []

    for f in manifest.files:
        f_lower = f.lower()
        if any(kw in f_lower for kw in keywords):
            found_files.append(f)

    evidence_details = []
    for f in found_files[:5]:
        evidence_details.append(EvidenceDetail(file_path=f, label="benchmark file"))

    if found_files:
        return True, Evidence(
            evidence_id="E-BENCH",
            type=EvidenceType.REPOSITORY_FILE,
            source="github",
            location=", ".join(found_files[:5]),
            content=f"Benchmark-related files found",
            supports=["BENCHMARK-OVERALL"],
        ), f"Benchmark files: {', '.join(found_files[:5])}", evidence_details

    return False, None, "No benchmark code found", evidence_details