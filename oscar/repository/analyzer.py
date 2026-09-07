"""Repository analyzer - static analysis of code structure.

Generates RepositoryManifest with classes, functions, line numbers, imports, etc.
"""

import ast
import os
from pathlib import Path
from typing import Optional

from oscar.models.schemas import CodeLocation, CodeSummary, RepositoryManifest
from oscar.utils.file_utils import find_python_files, read_file


def generate_manifest(repo_path: str, llm_summarize: bool = False) -> RepositoryManifest:
    """Generate a comprehensive repository manifest."""
    manifest = RepositoryManifest()
    repo_path = str(Path(repo_path).resolve())

    # Collect all files
    for root, dirs, files in os.walk(repo_path):
        # Skip hidden directories and common non-source dirs
        dirs[:] = [d for d in dirs if not d.startswith((".", "__pycache__", "node_modules", "venv", "env"))]

        rel_root = os.path.relpath(root, repo_path)
        if rel_root == ".":
            rel_root = ""

        for f in files:
            rel_path = os.path.join(rel_root, f) if rel_root else f
            manifest.files.append(rel_path)

            ext = os.path.splitext(f)[1].lower()
            if ext == ".py":
                manifest.python_modules.append(rel_path)
                _analyze_python_file(os.path.join(root, f), manifest, rel_path)
            elif ext in (".yaml", ".yml", ".toml", ".cfg", ".ini", ".json"):
                manifest.configs.append(rel_path)
            elif ext in (".sh", ".bat", ".ps1"):
                manifest.scripts.append(rel_path)

    # Detect dependencies
    manifest.dependencies = _detect_dependencies(repo_path)

    # Detect licenses
    manifest.licenses = _detect_licenses(repo_path)

    # Detect entry points
    manifest.entry_points = _detect_entry_points(repo_path, manifest)

    return manifest


def _analyze_python_file(file_path: str, manifest: RepositoryManifest, rel_path: str):
    """Analyze a Python file using AST and update manifest."""
    content = read_file(file_path)
    if content is None:
        return

    try:
        tree = ast.parse(content, filename=file_path)
    except SyntaxError:
        return

    classes = []
    class_locations = []
    functions = []
    function_locations = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            classes.append(node.name)
            class_locations.append(CodeLocation(
                module_path=rel_path,
                line_start=node.lineno,
                line_end=node.end_lineno or node.lineno,
                class_name=node.name,
            ))
            # Check for methods inside class
            for item in ast.iter_child_nodes(node):
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    full_name = f"{node.name}.{item.name}"
                    functions.append(full_name)
                    function_locations.append(CodeLocation(
                        module_path=rel_path,
                        line_start=item.lineno,
                        line_end=item.end_lineno or item.lineno,
                        class_name=node.name,
                        function_name=item.name,
                    ))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node.name)
            function_locations.append(CodeLocation(
                module_path=rel_path,
                line_start=node.lineno,
                line_end=node.end_lineno or node.lineno,
                function_name=node.name,
            ))

    if classes:
        manifest.classes[rel_path] = classes
    if class_locations:
        manifest.class_locations[rel_path] = class_locations
    if functions:
        manifest.functions[rel_path] = functions
    if function_locations:
        manifest.function_locations[rel_path] = function_locations


def search_code_content(
    repo_path: str,
    patterns: list[str],
    manifest: Optional[RepositoryManifest] = None,
    max_results: int = 10,
) -> list[CodeLocation]:
    """Search file content for specific patterns, returning line-level locations."""
    results = []
    modules = manifest.python_modules if manifest else find_python_files(repo_path)

    for mod in modules:
        content = read_file(os.path.join(repo_path, mod))
        if not content:
            continue
        lines = content.split("\n")
        for line_idx, line in enumerate(lines, 1):
            for pattern in patterns:
                if pattern.lower() in line.lower():
                    # Determine enclosing class/function
                    cls_name = _find_enclosing_class(content, line_idx)
                    func_name = _find_enclosing_function(content, line_idx)
                    results.append(CodeLocation(
                        module_path=mod,
                        line_start=line_idx,
                        class_name=cls_name,
                        function_name=func_name,
                        snippet=line.strip()[:200],
                    ))
                    if len(results) >= max_results:
                        return results
                    break  # One pattern match per line
    return results


def _find_enclosing_class(content: str, line_number: int) -> Optional[str]:
    """Find the class that encloses a given line number."""
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if node.lineno <= line_number <= (node.end_lineno or node.lineno):
                    return node.name
    except SyntaxError:
        pass
    return None


def _find_enclosing_function(content: str, line_number: int) -> Optional[str]:
    """Find the function that encloses a given line number."""
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.lineno <= line_number <= (node.end_lineno or node.lineno):
                    return node.name
    except SyntaxError:
        pass
    return None


def _detect_dependencies(repo_path: str) -> list[str]:
    """Detect dependency files and extract package names."""
    deps = []
    dep_files = ["requirements.txt", "environment.yml", "environment.yaml",
                 "pyproject.toml", "setup.py", "setup.cfg", "Pipfile"]

    for dep_file in dep_files:
        path = os.path.join(repo_path, dep_file)
        if os.path.isfile(path):
            content = read_file(path)
            if content:
                deps.append(f"=== {dep_file} ===")
                lines = content.strip().split("\n")
                deps.extend(lines[:50])
    return deps


def _detect_licenses(repo_path: str) -> list[str]:
    """Detect license files in the repository."""
    licenses = []
    for name in ["LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING", "COPYING.txt"]:
        path = os.path.join(repo_path, name)
        if os.path.isfile(path):
            content = read_file(path)
            if content:
                first_line = content.strip().split("\n")[0] if content else ""
                licenses.append(f"{name}: {first_line}")
    return licenses


def _detect_entry_points(repo_path: str, manifest: RepositoryManifest) -> list[str]:
    """Detect entry points (main scripts, CLI entry points)."""
    entry_points = []

    entry_names = ["main.py", "train.py", "test.py", "eval.py", "evaluate.py",
                   "run.py", "inference.py", "demo.py", "app.py", "cli.py",
                   "predict.py", "setup.py"]

    for f in manifest.files:
        basename = os.path.basename(f)
        if basename in entry_names:
            entry_points.append(f)

    for mod in manifest.python_modules:
        content = read_file(os.path.join(repo_path, mod))
        if content and ('if __name__ == "__main__"' in content or "argparse" in content):
            if mod not in entry_points:
                entry_points.append(mod)

    # set 去重后必须排序:set 迭代序随进程 hash 种子变化,直接 list(set(..))
    # 会让 entry_points 顺序逐进程随机 → 后续「首个匹配」逻辑与证据行全链抖动。
    return sorted(set(entry_points))