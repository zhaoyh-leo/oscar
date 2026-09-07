"""Dataset / Checkpoint / External Resources Audit.

Checks if external resources are documented, linked, and their access status is clear.
Uses README content analysis for accurate URL and documentation detection.
"""

import os
import re
from typing import Optional

from oscar.models.schemas import (
    AuditFinding, AuditState, ClaimCategory, ClaimStatus,
    Evidence, EvidenceDetail, EvidenceType, RepositoryManifest,
)
from oscar.utils.file_utils import read_file


def audit_resources(state: AuditState) -> list[AuditFinding]:
    """Audit dataset, checkpoint, and external resource documentation."""
    findings = []
    manifest = state.repository_manifest
    repo_path = state.project.get("clone_path", "")
    readme = state.readme_content or ""

    if not manifest:
        return findings

    # Check datasets
    dataset_finding = _check_datasets(manifest, repo_path, readme)
    findings.append(dataset_finding)

    # Check checkpoints/pretrained models
    checkpoint_finding = _check_checkpoints(manifest, repo_path, readme)
    findings.append(checkpoint_finding)

    # Check external resources
    external_finding = _check_external(manifest, repo_path, readme)
    findings.append(external_finding)

    return findings


def _extract_urls_from_readme(readme: str) -> list[tuple[int, str]]:
    """Extract URLs from README with line numbers."""
    urls = []
    for i, line in enumerate(readme.split("\n"), 1):
        found = re.findall(r"https?://[^\s\"'<>)]+", line)
        for url in found:
            urls.append((i, url))
    return urls


def _check_datasets(manifest: RepositoryManifest, repo_path: str, readme: str) -> AuditFinding:
    """Check dataset documentation."""
    dataset_files = [f for f in manifest.files if "dataset" in f.lower() or "data" in f.lower()]

    # Extract dataset-related URLs from README
    readme_urls = _extract_urls_from_readme(readme)
    dataset_urls = []
    for line_no, url in readme_urls:
        if any(kw in url.lower() for kw in ["dataset", "data", "download", "huggingface", "huggingface.co"]):
            dataset_urls.append((line_no, url))

    details = []
    # Add dataset files as evidence
    for f in dataset_files[:5]:
        details.append(EvidenceDetail(file_path=f, label="dataset file"))
    # Add dataset URLs as evidence
    for line_no, url in dataset_urls[:5]:
        details.append(EvidenceDetail(
            file_path="README.md",
            line_number=line_no,
            snippet=url[:200],
            label="dataset download URL",
        ))

    if dataset_files or dataset_urls:
        if dataset_urls:
            return AuditFinding(
                claim_id="RES-DATASET",
                category=ClaimCategory.DATASET,
                statement="Dataset documentation",
                status=ClaimStatus.VERIFIED,
                confidence=0.9,
                evidence_summary=f"Dataset files: {len(dataset_files)}, Download URLs: {len(dataset_urls)}",
                evidence_details=details,
                explanation=f"Found {len(dataset_files)} dataset-related files and {len(dataset_urls)} download URLs in README.",
            )
        else:
            return AuditFinding(
                claim_id="RES-DATASET",
                category=ClaimCategory.DATASET,
                statement="Dataset documentation",
                status=ClaimStatus.INCOMPLETE,
                confidence=0.6,
                evidence_summary=f"Dataset files: {len(dataset_files)}, but no download URLs found",
                evidence_details=details,
                explanation=f"Found {len(dataset_files)} dataset-related files but no explicit download URLs.",
            )
    else:
        return AuditFinding(
            claim_id="RES-DATASET",
            category=ClaimCategory.DATASET,
            statement="Dataset documentation",
            status=ClaimStatus.UNCERTAIN,
            confidence=0.3,
            evidence_summary="No dataset files or download URLs found",
            explanation="No dataset-related files or download URLs detected.",
        )


def _check_checkpoints(manifest: RepositoryManifest, repo_path: str, readme: str) -> AuditFinding:
    """Check checkpoint/pretrained model documentation."""
    kw_files = [f for f in manifest.files if any(kw in f.lower() for kw in
                                                 ["checkpoint", "weight", "pretrained", "model_zoo", "ckpt", "model.pth"])]

    # Extract checkpoint/model URLs from README
    readme_urls = _extract_urls_from_readme(readme)
    ckpt_urls = []
    for line_no, url in readme_urls:
        if any(kw in url.lower() for kw in ["checkpoint", "weight", "pretrained", "model", "huggingface.co"]):
            ckpt_urls.append((line_no, url))

    details = []
    for f in kw_files[:5]:
        details.append(EvidenceDetail(file_path=f, label="checkpoint file"))
    for line_no, url in ckpt_urls[:5]:
        details.append(EvidenceDetail(
            file_path="README.md",
            line_number=line_no,
            snippet=url[:200],
            label="model download URL",
        ))

    if kw_files or ckpt_urls:
        return AuditFinding(
            claim_id="RES-CHECKPOINT",
            category=ClaimCategory.CHECKPOINT,
            statement="Checkpoint/pretrained model documentation",
            status=ClaimStatus.VERIFIED if ckpt_urls else ClaimStatus.INCOMPLETE,
            confidence=0.85 if ckpt_urls else 0.5,
            evidence_summary=f"Checkpoint files: {len(kw_files)}, URLs: {len(ckpt_urls)}",
            evidence_details=details,
            explanation=f"Found {len(kw_files)} checkpoint-related files and {len(ckpt_urls)} download URLs.",
        )
    else:
        return AuditFinding(
            claim_id="RES-CHECKPOINT",
            category=ClaimCategory.CHECKPOINT,
            statement="Checkpoint/pretrained model documentation",
            status=ClaimStatus.UNCERTAIN,
            confidence=0.3,
            evidence_summary="No checkpoint files or download URLs found",
            explanation="No checkpoint-related files or download URLs detected.",
        )


def _check_external(manifest: RepositoryManifest, repo_path: str, readme: str) -> AuditFinding:
    """Check external resource documentation."""
    ext_files = [f for f in manifest.files if any(kw in f.lower() for kw in
                                                   ["download", "external", "third_party", "prepare", "setup"])]

    # Extract all external links from README
    readme_urls = _extract_urls_from_readme(readme)

    details = []
    for f in ext_files[:5]:
        details.append(EvidenceDetail(file_path=f, label="external resource file"))
    for line_no, url in readme_urls[:10]:
        details.append(EvidenceDetail(
            file_path="README.md",
            line_number=line_no,
            snippet=url[:200],
            label="external link",
        ))

    if ext_files or readme_urls:
        return AuditFinding(
            claim_id="RES-EXTERNAL",
            category=ClaimCategory.EXTERNAL_RESOURCE,
            statement="External resource documentation",
            status=ClaimStatus.VERIFIED,
            confidence=0.8,
            evidence_summary=f"External resource files: {len(ext_files)}, README links: {len(readme_urls)}",
            evidence_details=details,
            explanation=f"Found {len(ext_files)} external resource files and {len(readme_urls)} external links in README.",
        )
    else:
        return AuditFinding(
            claim_id="RES-EXTERNAL",
            category=ClaimCategory.EXTERNAL_RESOURCE,
            statement="External resource documentation",
            status=ClaimStatus.UNCERTAIN,
            confidence=0.5,
            evidence_summary="No external resource references found",
            explanation="No external resource references detected.",
        )