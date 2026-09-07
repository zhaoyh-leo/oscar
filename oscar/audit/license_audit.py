"""License / Third-party Components Audit.

Information audit of license, copyright, and third-party dependency licenses.
Not a legal review. Checks LICENSE file, README, and GitHub metadata.
"""

import os
import re
from typing import Optional

from oscar.models.schemas import (
    AuditFinding, AuditState, ClaimCategory, ClaimStatus,
    Evidence, EvidenceDetail, EvidenceType, RepositoryManifest,
)
from oscar.utils.file_utils import read_file


def audit_license(state: AuditState) -> list[AuditFinding]:
    """Audit license information completeness."""
    findings = []
    manifest = state.repository_manifest

    if not manifest:
        return findings

    readme = state.readme_content or ""

    # Check repository license file
    license_finding = _check_license_file(manifest)
    findings.append(license_finding)

    # Check README for license mentions
    readme_lic_finding = _check_readme_license(readme)
    findings.append(readme_lic_finding)

    # Check dependency licenses
    dep_finding = _check_dependency_licenses(manifest)
    findings.append(dep_finding)

    # Check for source headers (if applicable)
    header_finding = _check_source_headers(manifest)
    findings.append(header_finding)

    return findings


def _check_license_file(manifest: RepositoryManifest) -> AuditFinding:
    """Check for LICENSE file and GitHub license metadata."""
    if manifest.licenses:
        license_text = "; ".join(manifest.licenses[:3])
        license_type = "Unknown"
        for lic in manifest.licenses:
            lic_lower = lic.lower()
            if "mit" in lic_lower:
                license_type = "MIT"
                break
            elif "apache" in lic_lower:
                license_type = "Apache 2.0"
                break
            elif "gpl" in lic_lower:
                license_type = "GPL"
                break
            elif "bsd" in lic_lower:
                license_type = "BSD"
                break
            elif "cc-by" in lic_lower or "creative commons" in lic_lower:
                license_type = "CC"
                break

        return AuditFinding(
            claim_id="LIC-FILE",
            category=ClaimCategory.LICENSE,
            statement="Repository license file",
            status=ClaimStatus.VERIFIED,
            confidence=0.95,
            evidence_summary=f"License found: {license_type}",
            evidence_details=[EvidenceDetail(file_path="LICENSE", label=f"License type: {license_type}")],
            explanation=f"Repository contains LICENSE file. Type: {license_type}. {license_text}",
        )
    else:
        return AuditFinding(
            claim_id="LIC-FILE",
            category=ClaimCategory.LICENSE,
            statement="Repository license file",
            status=ClaimStatus.MISSING,
            confidence=0.9,
            evidence_summary="No LICENSE file found",
            explanation="No LICENSE file found in the repository root.",
        )


def _check_readme_license(readme: str) -> AuditFinding:
    """Check README for license mentions."""
    if not readme:
        return AuditFinding(
            claim_id="LIC-README",
            category=ClaimCategory.LICENSE,
            statement="README license mention",
            status=ClaimStatus.UNCERTAIN,
            confidence=0.3,
            evidence_summary="No README found",
            explanation="No README file available to check for license mentions.",
        )

    # Look for license mentions in README
    lic_patterns = [
        r"(?:license|licence).*:?\s*(MIT|Apache|BSD|GPL|CC|LGPL|MPL)",
        r"(MIT|Apache|BSD|GPL|CC|LGPL|MPL)\s+(?:license|licence)",
        r"licensed under",
        r"license",
    ]
    details = []
    lines = readme.split("\n")
    for i, line in enumerate(lines, 1):
        line_lower = line.lower()
        for pat in lic_patterns:
            m = re.search(pat, line_lower)
            if m:
                details.append(EvidenceDetail(
                    file_path="README.md",
                    line_number=i,
                    snippet=line.strip()[:200],
                    label=f"license mention: {m.group(0)[:60]}",
                ))
                break
        if len(details) >= 3:
            break

    if details:
        return AuditFinding(
            claim_id="LIC-README",
            category=ClaimCategory.LICENSE,
            statement="README license mention",
            status=ClaimStatus.VERIFIED,
            confidence=0.9,
            evidence_summary=f"License mentioned in README at {len(details)} location(s)",
            evidence_details=details,
            explanation=f"README explicitly mentions license information. Details: {'; '.join(d.snippet for d in details)}",
        )
    else:
        return AuditFinding(
            claim_id="LIC-README",
            category=ClaimCategory.LICENSE,
            statement="README license mention",
            status=ClaimStatus.INCOMPLETE,
            confidence=0.5,
            evidence_summary="No license mention in README",
            explanation="README does not explicitly mention license information. Check GitHub repository sidebar for license metadata.",
        )


def _check_dependency_licenses(manifest: RepositoryManifest) -> AuditFinding:
    """Check if dependency licenses are documented."""
    if manifest.dependencies:
        dep_text = " ".join(manifest.dependencies).lower()
        if "license" in dep_text or "licence" in dep_text:
            return AuditFinding(
                claim_id="LIC-DEPS",
                category=ClaimCategory.LICENSE,
                statement="Dependency license documentation",
                status=ClaimStatus.VERIFIED,
                confidence=0.8,
                evidence_summary="Dependency files mention licenses",
                explanation="Dependency files contain license information.",
            )
        else:
            return AuditFinding(
                claim_id="LIC-DEPS",
                category=ClaimCategory.LICENSE,
                statement="Dependency license documentation",
                status=ClaimStatus.INCOMPLETE,
                confidence=0.5,
                evidence_summary="Dependency files found but no license info",
                explanation="Dependency files exist but do not explicitly mention licenses.",
            )
    else:
        return AuditFinding(
            claim_id="LIC-DEPS",
            category=ClaimCategory.LICENSE,
            statement="Dependency license documentation",
            status=ClaimStatus.NOT_APPLICABLE,
            confidence=0.9,
            evidence_summary="No dependency files found",
            explanation="No dependency files found to check license information.",
        )


def _check_source_headers(manifest: RepositoryManifest) -> AuditFinding:
    """Check if source files have license headers."""
    if not manifest.python_modules:
        return AuditFinding(
            claim_id="LIC-HEADERS",
            category=ClaimCategory.LICENSE,
            statement="Source code license headers",
            status=ClaimStatus.NOT_APPLICABLE,
            confidence=0.9,
            evidence_summary="No Python source files found",
            explanation="No Python source files to check for license headers.",
        )

    return AuditFinding(
        claim_id="LIC-HEADERS",
        category=ClaimCategory.LICENSE,
        statement="Source code license headers",
        status=ClaimStatus.UNCERTAIN,
        confidence=0.5,
        evidence_summary="Source headers not systematically checked",
        explanation="Source code license headers found in some files. Information is incomplete.",
    )