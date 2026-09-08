"""Dataset / Checkpoint / External Resources / Release Delivery Audit.

Checks if external resources are documented, linked, and their access status is clear.
Uses README content analysis for accurate URL and documentation detection.

模块级检查(RES-DATASET / RES-CHECKPOINT / RES-EXTERNAL)之外,本节点消费
「交付承诺」类 claim(release / implementation / dataset / checkpoint):
论文/README 声称将发布代码/权重/数据集 → 按仓库物证(代码模块、权重文件
与 README 下载 URL、数据集文件与 URL)裁决为 RELEASE 类 finding。此前这
类 claim 抽取后无人消费:不产生 finding、不进 issue 调查,「代码/权重将
发布」这类最常不兑现的承诺在报告中完全无痕。
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

    # Delivery-promise claims (release / implementation / dataset / checkpoint)
    findings.extend(_audit_delivery_claims(state, manifest, readme))

    return findings


def _extract_urls_from_readme(readme: str) -> list[tuple[int, str]]:
    """Extract URLs from README with line numbers."""
    urls = []
    for i, line in enumerate(readme.split("\n"), 1):
        found = re.findall(r"https?://[^\s\"'<>)]+", line)
        for url in found:
            urls.append((i, url))
    return urls


# 文件路径/README URL 关键词(交付物证;模块级 RES-* 检查与新交付审计共用)
_DATASET_FILE_KWS = ("dataset", "data")
_DATASET_URL_KWS = ("dataset", "data", "download", "huggingface", "huggingface.co")
_CKPT_FILE_KWS = (
    "checkpoint", "weight", "pretrained", "model_zoo", "ckpt",
    "model.pth", "safetensors",
)
_CKPT_URL_KWS = (
    "checkpoint", "weight", "pretrained", "model", "huggingface.co",
    "hf.co", "zenodo", "drive.google", "pan.baidu", "releases",
)


def _urls_matching(readme: str, keywords: tuple[str, ...]) -> list[tuple[int, str]]:
    """README 中含任一关键词的 URL(带行号)。"""
    out = []
    for line_no, url in _extract_urls_from_readme(readme):
        if any(kw in url.lower() for kw in keywords):
            out.append((line_no, url))
    return out


def _check_datasets(manifest: RepositoryManifest, repo_path: str, readme: str) -> AuditFinding:
    """Check dataset documentation."""
    dataset_files = [f for f in manifest.files if any(k in f.lower() for k in _DATASET_FILE_KWS)]

    # Extract dataset-related URLs from README
    dataset_urls = _urls_matching(readme, _DATASET_URL_KWS)

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
    kw_files = [f for f in manifest.files if any(kw in f.lower() for kw in _CKPT_FILE_KWS)]

    # Extract checkpoint/model URLs from README
    ckpt_urls = _urls_matching(readme, _CKPT_URL_KWS)

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


# ---------------------------------------------------------------------------
# Delivery-promise audit(release / implementation / dataset / checkpoint claims)
# ---------------------------------------------------------------------------

# 交付承诺类 claim 在此消费。core_method / api_interface / benchmark 的
# claim 有各自的逐条审计主;license 由 license_audit 规则检查覆盖;均不
# 在此重复审计。
_DELIVERY_CATEGORIES = {
    ClaimCategory.RELEASE,
    ClaimCategory.IMPLEMENTATION,  # 抽取端已折入 RELEASE;此处兜底旧路径
    ClaimCategory.DATASET,         # "dataset X is constructed" → 数据交付物
    ClaimCategory.CHECKPOINT,      # "pretrained weights released" → 权重物证
}

# claim 语句 → 承诺交付物 分型词(statement 为英文,小写匹配)
_WEIGHT_TERMS = ("weight", "checkpoint", "pretrained", "ckpt")
_DATA_TERMS = ("dataset", "corpus")


def _audit_delivery_claims(
    state: AuditState,
    manifest: RepositoryManifest,
    readme: str,
) -> list[AuditFinding]:
    """Audit delivery-promise claims against repository artifacts.

    「代码/权重/数据集将发布」——开源审计里最常不兑现的一类承诺。此前抽
    取后无任何节点消费:不产 finding、不进 issue 调查(作者「稍后发布」的
    PLANNED 证据链断掉)、报告完全无痕。此处按承诺的交付物找仓库物证:
    code → python 模块;weights → 权重文件 + README 下载 URL;dataset →
    数据集文件 + README URL。产出 RELEASE 类 finding(不计分,列于
    Informational);MISSING/INCOMPLETE 因而进入 issue 调查 → 作者显式
    「稍后发布」落为 PLANNED,形成闭环。
    """
    claims = [
        c for c in (state.claims or [])
        if c.category in _DELIVERY_CATEGORIES
    ]
    if not claims:
        return []

    findings = []
    for claim in claims:
        kinds = {"code"}
        text = (claim.statement or "").lower()
        if any(w in text for w in _WEIGHT_TERMS):
            kinds.add("weights")
        if any(w in text for w in _DATA_TERMS):
            kinds.add("dataset")

        details: list[EvidenceDetail] = []
        hits: list[str] = []
        for kind in sorted(kinds):
            ev_details = _delivery_evidence(kind, manifest, readme)
            if ev_details:
                hits.append(kind)
            details.extend(ev_details)

        if len(hits) == len(kinds):
            status, confidence = ClaimStatus.VERIFIED, 0.85
        elif hits:
            status, confidence = ClaimStatus.INCOMPLETE, 0.6
        else:
            status, confidence = ClaimStatus.MISSING, 0.4

        promised = ", ".join(sorted(kinds))
        findings.append(AuditFinding(
            claim_id=claim.claim_id,
            category=ClaimCategory.RELEASE,
            statement=claim.statement,
            status=status,
            confidence=confidence,
            evidence_summary=(
                f"Promised: {promised}. Artifacts found: {', '.join(hits) or 'none'}"
            ),
            evidence_details=details,
            explanation=(
                f"The claim promises delivery of: {promised}. "
                + (
                    f"Repository artifacts confirm: {', '.join(hits)}."
                    if hits else
                    "No repository artifacts match the promised deliverables."
                )
            ),
        ))
    return findings


def _delivery_evidence(
    kind: str, manifest: RepositoryManifest, readme: str
) -> list[EvidenceDetail]:
    """Return repository artifact evidence for one promised deliverable."""
    details: list[EvidenceDetail] = []
    if kind == "code":
        for f in manifest.python_modules[:5]:
            details.append(EvidenceDetail(file_path=f, label="code module"))
        return details
    if kind == "weights":
        for f in [
            f for f in manifest.files
            if any(k in f.lower() for k in _CKPT_FILE_KWS)
        ][:5]:
            details.append(EvidenceDetail(file_path=f, label="checkpoint/weight file"))
        for line_no, url in _urls_matching(readme, _CKPT_URL_KWS)[:5]:
            details.append(EvidenceDetail(
                file_path="README.md", line_number=line_no,
                snippet=url[:200], label="model download URL",
            ))
        return details
    # dataset
    for f in [
        f for f in manifest.files
        if any(k in f.lower() for k in _DATASET_FILE_KWS)
    ][:5]:
        details.append(EvidenceDetail(file_path=f, label="dataset file"))
    for line_no, url in _urls_matching(readme, _DATASET_URL_KWS)[:5]:
        details.append(EvidenceDetail(
            file_path="README.md", line_number=line_no,
            snippet=url[:200], label="dataset download URL",
        ))
    return details