"""Training Completeness Audit.

Checks if training code (dataset, loss, optimizer, training loop, etc.) is complete.
Uses content-level analysis (not just filename matching) to detect components.
"""

import ast
import os
from typing import Optional

from oscar.models.schemas import (
    AuditFinding, AuditState, ClaimCategory, ClaimStatus,
    CodeLocation, Evidence, EvidenceDetail, EvidenceType, RepositoryManifest,
)
from oscar.repository.analyzer import search_code_content
from oscar.utils.file_utils import read_file


def audit_training(state: AuditState) -> list[AuditFinding]:
    """Audit training code completeness.

    Checks for: Dataset Loader, Loss, Optimizer, Scheduler,
    Training Loop, Checkpoint Saving, Configuration, Entry Point.

    Uses content-level analysis (AST + regex) to find components
    even when filenames don't match expected keywords.
    """
    findings = []
    manifest = state.repository_manifest
    repo_path = state.project.get("clone_path", "")

    if not manifest:
        return findings

    components = {
        "Dataset Loader": _check_dataset_loader(manifest, repo_path),
        "Loss Function": _check_loss(manifest, repo_path),
        "Optimizer": _check_optimizer(manifest, repo_path),
        "Learning Rate Scheduler": _check_scheduler(manifest, repo_path),
        "Training Loop": _check_training_loop(repo_path, manifest),
        "Checkpoint Saving": _check_checkpoint(manifest, repo_path),
        "Configuration": _check_config(manifest),
        "Entry Point": _check_entry_point(manifest),
    }

    found = [k for k, v in components.items() if v["found"]]
    missing = [k for k, v in components.items() if not v["found"]]
    found_count = len(found)
    total = len(components)

    # Collect all evidence details
    all_details: list[EvidenceDetail] = []
    for comp_name, comp_data in components.items():
        all_details.extend(comp_data.get("details", []))

    if found_count == total:
        status = ClaimStatus.VERIFIED
        confidence = 0.95
    elif found_count >= total * 0.6:
        status = ClaimStatus.INCOMPLETE
        confidence = 0.7 + 0.25 * (found_count / total)
    elif found_count > 0:
        status = ClaimStatus.INCOMPLETE
        confidence = 0.5
    else:
        status = ClaimStatus.MISSING
        confidence = 0.2

    # Build detailed evidence table string
    evidence_lines = []
    evidence_lines.append(f"| Component | Found | Location | Line | Details |")
    evidence_lines.append(f"|-----------|-------|----------|------|---------|")
    for comp_name, comp_data in components.items():
        if comp_data["found"]:
            first = comp_data["details"][0] if comp_data["details"] else None
            if first:
                loc = first.file_path
                line = str(first.line_number) if first.line_number else "-"
                detail = first.label or first.snippet[:40]
            else:
                loc, line, detail = "-", "-", "-"
            evidence_lines.append(f"| {comp_name} | YES | {loc} | {line} | {detail} |")
        else:
            evidence_lines.append(f"| {comp_name} | NO | - | - | - |")

    evidence_table = "\n".join(evidence_lines)

    finding = AuditFinding(
        claim_id="TRAINING-OVERALL",
        category=ClaimCategory.TRAINING,
        statement="Training code completeness",
        status=status,
        confidence=confidence,
        evidence_summary=f"Found {found_count}/{total} training components. Missing: {', '.join(missing)}" if missing else f"All {total} training components found.",
        evidence_details=all_details,
        explanation=f"Training components found: {', '.join(found)}. Missing: {', '.join(missing)}.\n\nEvidence Table:\n{evidence_table}",
    )
    findings.append(finding)

    return findings


def _check_dataset_loader(manifest: RepositoryManifest, repo_path: str) -> dict:
    """Check for dataset/dataloader by filename + content."""
    # Filename-level check
    for f in manifest.files:
        f_lower = f.lower()
        if "dataset" in f_lower or "dataloader" in f_lower or "data_loader" in f_lower:
            return _make_found(f, label="dataset file")

    # Class/function name check
    for mod, classes in manifest.classes.items():
        for cls in classes:
            if any(kw in cls.lower() for kw in ["dataset", "dataloader", "data_loader"]):
                return _make_found(mod, cls, label=f"class {cls}")

    # Content-level: search for Dataset/DataLoader class usage
    details = search_code_content(repo_path, [
        "class.*Dataset", "torch.utils.data", "DataLoader",
        "Dataset(", "ImageFolder", "DataLoader(",
    ], manifest, max_results=5)
    if details:
        return _make_found_details(details, label="dataset reference")

    return _not_found()


def _check_loss(manifest: RepositoryManifest, repo_path: str) -> dict:
    """Check for loss function definition/usage."""
    for f in manifest.files:
        if "loss" in f.lower():
            return _make_found(f, label="loss file")

    for mod, funcs in manifest.functions.items():
        for func in funcs:
            if "loss" in func.lower():
                return _make_found(mod, func, label=f"function {func}")

    # Content-level: search for loss definitions
    details = search_code_content(repo_path, [
        "nn.*Loss", "loss_fn", "criterion", "loss_func",
        "loss_function", "def.*loss", "F.*loss",
        "CrossEntropyLoss", "MSELoss", "L1Loss", "BCELoss",
    ], manifest, max_results=5)
    if details:
        return _make_found_details(details, label="loss reference")

    return _not_found()


def _check_optimizer(manifest: RepositoryManifest, repo_path: str) -> dict:
    """Check for optimizer definition. Content-level check is critical."""
    for f in manifest.files:
        if "optim" in f.lower() or "adam" in f.lower() or "sgd" in f.lower():
            return _make_found(f, label="optimizer file")

    for mod, funcs in manifest.functions.items():
        for func in funcs:
            if "optim" in func.lower():
                return _make_found(mod, func, label=f"function {func}")

    # Content-level: search for optimizer variable assignments
    # This catches patterns like: optimizer = Adam(...), optimizer = torch.optim.Adam(...)
    details = search_code_content(repo_path, [
        "optimizer", "optim =", "optimizer =",
        "Adam(", "SGD(", "AdamW(", "RMSprop(",
        "torch.optim", "optim.Adam", "optim.SGD",
    ], manifest, max_results=5)
    if details:
        return _make_found_details(details, label="optimizer reference")

    return _not_found()


def _check_scheduler(manifest: RepositoryManifest, repo_path: str) -> dict:
    """Check for learning rate scheduler."""
    for f in manifest.files:
        if "scheduler" in f.lower() or "lr" in f.lower():
            return _make_found(f, label="scheduler file")

    # Content-level
    details = search_code_content(repo_path, [
        "scheduler", "lr_scheduler", "StepLR", "MultiStepLR",
        "CosineAnnealing", "ReduceLROnPlateau", "ExponentialLR",
        "LambdaLR", "lr =", "learning_rate",
    ], manifest, max_results=5)
    if details:
        return _make_found_details(details, label="scheduler/lr reference")

    return _not_found()


def _check_training_loop(repo_path: str, manifest: RepositoryManifest) -> dict:
    """Check for training loop patterns."""
    for mod in manifest.python_modules:
        content = read_file(os.path.join(repo_path, mod))
        if not content:
            continue
        lines = content.split("\n")
        patterns = [
            "for epoch", "for batch", "for data",
            "train_loader", "training_loop",
            "model.train()", "model.train(",
            "backward()", "loss.backward",
            "optimizer.step", "optim.step",
        ]
        for line_idx, line in enumerate(lines, 1):
            for pat in patterns:
                if pat in line.lower():
                    return _make_found_details([
                        EvidenceDetail(
                            file_path=mod,
                            line_number=line_idx,
                            snippet=line.strip()[:200],
                            label=f"training loop: {pat}",
                        )
                    ], label="training loop")
    return _not_found()


def _check_checkpoint(manifest: RepositoryManifest, repo_path: str) -> dict:
    """Check for checkpoint saving/loading."""
    # Filename
    for f in manifest.files:
        if any(kw in f.lower() for kw in ["checkpoint", "ckpt", "save", "model_zoo"]):
            return _make_found(f, label="checkpoint file")

    # Content
    details = search_code_content(repo_path, [
        "torch.save", "torch.load", "checkpoint", "save_checkpoint",
        "load_checkpoint", "model.save", "model.load",
        "save_model", "load_model", "state_dict",
    ], manifest, max_results=5)
    if details:
        return _make_found_details(details, label="checkpoint reference")

    return _not_found()


def _check_config(manifest: RepositoryManifest) -> dict:
    """Check for configuration files."""
    if manifest.configs:
        return _make_found(manifest.configs[0], label="config file")
    # Also check for argparse
    for mod, funcs in manifest.functions.items():
        for func in funcs:
            if "argparse" in func.lower() or "config" in func.lower():
                return _make_found(mod, func, label=f"function {func}")
    return _not_found()


def _check_entry_point(manifest: RepositoryManifest) -> dict:
    """Check for training entry point."""
    train_files = [f for f in manifest.entry_points if "train" in f.lower()]
    if train_files:
        return _make_found(train_files[0], label="train entry point")
    return _not_found()


def _make_found(file_path: str, class_or_func: str = "", label: str = "") -> dict:
    details = [EvidenceDetail(file_path=file_path, label=label or class_or_func)]
    return {"found": True, "details": details}


def _make_found_details(details: list, label: str = "") -> dict:
    """Convert mixed detail types to EvidenceDetail list.

    Accepts list of CodeLocation or EvidenceDetail, converting CodeLocation
    to EvidenceDetail automatically.
    """
    converted = []
    for d in details:
        if isinstance(d, CodeLocation):
            converted.append(EvidenceDetail(
                file_path=d.module_path,
                line_number=d.line_start,
                class_name=d.class_name,
                function_name=d.function_name,
                snippet=d.snippet,
                label=label,
            ))
        elif isinstance(d, EvidenceDetail):
            converted.append(d)
        else:
            converted.append(EvidenceDetail(
                file_path=str(getattr(d, "file_path", "")),
                label=label,
            ))
    return {"found": True, "details": converted}


def _not_found() -> dict:
    return {"found": False, "details": []}