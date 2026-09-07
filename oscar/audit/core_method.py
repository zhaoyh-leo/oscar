"""Core Method Completeness Audit.

Checks whether paper-described core methods have corresponding implementations.
"""

from typing import Optional

from oscar.mapping.claim_code_mapper import map_claim_to_code
from oscar.models.schemas import (
    AuditFinding, AuditState, Claim, ClaimCategory, ClaimStatus,
    ClaimSource, Evidence, EvidenceDetail, EvidenceType, MappingResult, RepositoryManifest,
)


def audit_core_methods(state: AuditState) -> list[AuditFinding]:
    """Audit core method completeness."""
    findings = []
    manifest = state.repository_manifest
    repo_path = state.project.get("clone_path", "")

    if not manifest:
        return findings

    # Get core method claims
    method_claims = [c for c in state.claims if c.category == ClaimCategory.CORE_METHOD]

    # If no method claims from paper, generate from manifest
    if not method_claims and manifest.classes:
        method_claims = _generate_method_claims_from_manifest(manifest)

    for claim in method_claims:
        mapping = map_claim_to_code(claim, manifest, repo_path)

        evidence = _build_mapping_evidence(mapping, repo_path)
        claim.repository_evidence = [evidence] if evidence else []

        if mapping.decision == "MATCH":
            if mapping.confidence >= 0.7:
                status = ClaimStatus.VERIFIED
            else:
                status = ClaimStatus.INCOMPLETE
        elif mapping.confidence > 0:
            status = ClaimStatus.INCOMPLETE
        else:
            status = ClaimStatus.MISSING

        # Build evidence_details from mapping result (rows feed the grounder's
        # structured recall pass — mapper files must not be truncated to noise)
        evidence_details = []
        for f in mapping.candidate_files[:12]:
            evidence_details.append(EvidenceDetail(
                file_path=f,
                label="candidate file",
            ))
        # 不再产出 candidate class/function 行:它们只带名字、无属主文件/行号
        # (报告里 File/Line 全空的裸行)。真实的类/方法锚定由 grounding 阶段
        # 以 chunk 记录给出(file+line+symbol+代码摘要);结构化召回只需
        # 候选文件行(chunks_for_files 按文件取类级代码块)。

        finding = AuditFinding(
            claim_id=claim.claim_id,
            category=ClaimCategory.CORE_METHOD,
            statement=claim.statement,
            status=status,
            confidence=mapping.confidence,
            evidence_summary=f"Files: {', '.join(mapping.candidate_files[:5])}" if mapping.candidate_files else "No matching files found",
            evidence_details=evidence_details,
            explanation=mapping.reason,
        )
        findings.append(finding)

    return findings


def _generate_method_claims_from_manifest(manifest: RepositoryManifest) -> list[Claim]:
    """Generate method claims from repository manifest classes."""
    claims = []
    counter = 1

    for module_path, classes in manifest.classes.items():
        for cls in classes:
            # Skip common utility classes
            if cls.lower() in ("config", "utils", "logger", "base", "loss"):
                continue
            claim = Claim(
                claim_id=f"METHOD-M{counter:03d}",
                category=ClaimCategory.CORE_METHOD,
                statement=cls,  # Use the actual class name
                source=ClaimSource(type="repository", location=module_path, content=f"class {cls}"),
            )
            claims.append(claim)
            counter += 1
            if counter > 20:
                break
        if counter > 20:
            break

    return claims


def _build_mapping_evidence(mapping: MappingResult, repo_path: str) -> Optional[Evidence]:
    """Build evidence from mapping result."""
    if not mapping.candidate_files:
        return None

    return Evidence(
        evidence_id=f"E-MAP-{mapping.claim_id}",
        type=EvidenceType.REPOSITORY_FILE,
        source="github",
        location=", ".join(mapping.candidate_files[:5]),
        content=f"Mapping result: {mapping.reason}. Files: {', '.join(mapping.candidate_files[:5])}",
        supports=[mapping.claim_id],
    )