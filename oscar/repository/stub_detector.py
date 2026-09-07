"""Stub detection - find placeholder/incomplete implementations."""

import ast
import os
from typing import Optional

from oscar.utils.file_utils import read_file, find_python_files


def detect_stubs(repo_path: str) -> dict[str, list[str]]:
    """Detect stub implementations in Python files.

    Returns a dict mapping file paths to list of stub descriptions.
    """
    stub_details = {}
    python_files = find_python_files(repo_path)

    for file_path in python_files:
        stubs = _analyze_stubs(file_path)
        if stubs:
            rel_path = os.path.relpath(file_path, repo_path)
            stub_details[rel_path] = stubs

    return stub_details


def _analyze_stubs(file_path: str) -> list[str]:
    """Analyze a single Python file for stub patterns."""
    content = read_file(file_path)
    if content is None:
        return []

    stubs = []

    # Check for text-based patterns (works for non-Python files too)
    stub_patterns = [
        "TODO", "FIXME", "NotImplemented", "stub", "placeholder",
        "not implemented", "to be implemented", "coming soon",
    ]
    content_lower = content.lower()
    for pattern in stub_patterns:
        for i, line in enumerate(content.split("\n"), 1):
            if pattern.lower() in line.lower() and not line.strip().startswith("#"):
                stubs.append(f"Line {i}: {line.strip()[:100]}")
                if len(stubs) >= 3:  # Max 3 per pattern per file
                    break

    # AST-based analysis for Python-specific stubs
    try:
        tree = ast.parse(content, filename=file_path)
        for node in ast.walk(tree):
            # Check for empty function bodies
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # Check if body is just pass, ..., or raise NotImplementedError
                body = node.body
                if len(body) == 1:
                    if isinstance(body[0], ast.Pass):
                        stubs.append(f"Empty function/class: {node.name} (pass)")
                    elif isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Ellipsis):
                        stubs.append(f"Empty function/class: {node.name} (...)")
                    elif isinstance(body[0], ast.Raise):
                        if hasattr(body[0], "exc") and body[0].exc:
                            if isinstance(body[0].exc, ast.Call) and hasattr(body[0].exc, "func"):
                                if getattr(body[0].exc.func, "id", "") == "NotImplementedError":
                                    stubs.append(f"Stub function: {node.name} (raises NotImplementedError)")
    except SyntaxError:
        pass

    return sorted(set(stubs))[:10]  # Max 10 per file; sorted → 确定性顺序