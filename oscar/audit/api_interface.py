"""API / Interface Completeness Audit.

Checks if declared APIs (from README, docs, etc.) have corresponding implementations.
"""

import os
import re
from typing import Optional

from oscar.models.schemas import (
    AuditFinding, AuditState, ClaimCategory, ClaimStatus,
    Evidence, EvidenceDetail, EvidenceType, RepositoryManifest,
)
from oscar.utils.file_utils import read_file
from oscar.repository.stub_detector import detect_stubs


def audit_api_interface(state: AuditState) -> list[AuditFinding]:
    """Audit API/interface completeness."""
    findings = []
    manifest = state.repository_manifest
    repo_path = state.project.get("clone_path", "")

    if not manifest:
        return findings

    # Check for stub implementations
    stub_details = detect_stubs(repo_path) if repo_path else {}

    total_stubs = sum(len(v) for v in stub_details.values())
    stub_files = list(stub_details.keys())

    # Check for declared API claims
    api_claims = [c for c in state.claims if c.category == ClaimCategory.API_INTERFACE]

    if api_claims:
        for claim in api_claims:
            # Try to map the claim to code
            claim_keywords = re.findall(r'[a-zA-Z_][a-zA-Z0-9_.]*', claim.statement.lower())
            matched = False
            for kw in claim_keywords:
                for f in manifest.files:
                    if kw in f.lower():
                        matched = True
                        break
                if matched:
                    break

            if matched and total_stubs == 0:
                status = ClaimStatus.VERIFIED
                confidence = 0.85
            elif matched and total_stubs > 0:
                status = ClaimStatus.INCOMPLETE
                confidence = 0.6
            else:
                status = ClaimStatus.MISSING
                confidence = 0.3

            evidence_details = []
            for f in stub_files[:5]:
                evidence_details.append(EvidenceDetail(file_path=f, label="stub file"))

            finding = AuditFinding(
                claim_id=claim.claim_id,
                category=ClaimCategory.API_INTERFACE,
                statement=claim.statement,
                status=status,
                confidence=confidence,
                evidence_summary=f"Stub files: {len(stub_files)}" if stub_files else "No stubs detected",
                evidence_details=evidence_details,
                explanation=f"Claim: {claim.statement}. Stubs: {total_stubs} in {len(stub_files)} files." if total_stubs > 0 else f"Claim: {claim.statement}. No stubs found.",
            )
            findings.append(finding)
    else:
        # General API completeness check
        api_files = [f for f in manifest.files if any(kw in f.lower() for kw in ["api", "cli", "main", "cli", "interface"])]
        has_api = len(api_files) > 0

        if has_api and total_stubs == 0:
            status = ClaimStatus.VERIFIED
            confidence = 0.8
        elif has_api and total_stubs > 0:
            status = ClaimStatus.INCOMPLETE
            confidence = 0.5
        else:
            # 仓库不存在 API/接口面:是「不适用」而非「满足」——不存在 ≠ 扣 0 分,
            # 也不会被误计为 VERIFIED 抬高分数(该语义由计分口径处理)。
            status = ClaimStatus.NOT_APPLICABLE
            confidence = 0.9

        evidence_details = []
        for f in api_files[:5]:
            evidence_details.append(EvidenceDetail(file_path=f, label="api file"))
        for f in stub_files[:5]:
            evidence_details.append(EvidenceDetail(file_path=f, label="stub file"))

        if status == ClaimStatus.NOT_APPLICABLE:
            explanation = (
                "No API/interface surface found in the repository (no api/cli/"
                "interface files); this audit dimension does not apply."
            )
        else:
            explanation = (
                f"Found {len(api_files)} API-related files, {total_stubs} stub "
                f"instances across {len(stub_files)} files."
            )

        finding = AuditFinding(
            claim_id="API-OVERALL",
            category=ClaimCategory.API_INTERFACE,
            statement="API/Interface completeness",
            status=status,
            confidence=confidence,
            evidence_summary=f"API files: {len(api_files)}, Stubs: {total_stubs}" if api_files else "No explicit API files found",
            evidence_details=evidence_details,
            explanation=explanation,
        )
        findings.append(finding)

    return findings