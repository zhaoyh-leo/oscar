"""Claim → Code mapping (candidate pre-screening).

把 claim 映射到候选代码,多策略从低价(名字精确匹配)到高价(hybrid 检索)。
本模块只做**召回候选**,不做最终裁决:

- 旧版策略 4b(GraphCodeBERT 结构审查)与策略 5(LLM 语义映射)已移除:
  4b 只审第一个候选文件的文件头、其统计文本会泄漏进报告 evidence;
  5 是不含代码原文的弱版裁决。两者被 evidence_grounder 取代——它把真正的
  代码块原文(函数/类体)喂给 LLM,产出定位 + 自然语言摘要 + 状态裁决。

符号命中按属主 module 归属:类/函数遍历的是 manifest.classes.items()/
functions.items()(module → names),命中即记到拥有该符号的模块,不再出现
「外层遍历文件、内层遍历所有文件名字」导致的张冠李戴。
"""

import os
import re

from oscar.models.schemas import (
    Claim, ClaimCategory, MappingResult, RepositoryManifest,
)
from oscar.mapping.code_vector_store import code_vector_store
from oscar.paper.analyzer import claim_search_text


def map_claim_to_code(
    claim: Claim,
    manifest: RepositoryManifest,
    repo_path: str,
) -> MappingResult:
    """Map a claim to candidate code. Never raises; returns a MappingResult.

    Strategies (candidate recall only):
    1. Exact filename match
    2. Exact class/function name match (attributed to owning module)
    3. Partial filename match
    4. Category file patterns
    5. Hybrid retrieval (BM25 + Vector + Keyword, fused via RRF)
    """
    result = MappingResult(claim_id=claim.claim_id)
    claim_text = claim.statement

    for keyword in _extract_keywords(claim_text):
        keyword_lower = keyword.lower()

        # Strategy 1: exact filename match (basename without extension)
        for f in manifest.files:
            fname = os.path.basename(f).lower()
            fname_no_ext = os.path.splitext(fname)[0]
            if keyword_lower == fname_no_ext:
                _add_file(result, f)
                if result.decision == "UNMATCHED" or result.confidence < 0.9:
                    result.decision = "MATCH"
                    result.confidence = max(result.confidence, 0.9)
                    result.reason = f"Exact filename match: {f}"

        # Strategy 2: exact class / function name match, attributed to the
        # module that actually owns the symbol (张冠李戴修复)。
        for mod, cls_list in manifest.classes.items():
            if any(keyword_lower == c.lower() for c in cls_list):
                _add_file(result, mod)
                result.candidate_classes.append(keyword)
                result.decision = "MATCH"
                result.confidence = max(result.confidence, 0.85)
                result.reason = f"Class name match: {keyword} in {mod}"

        for mod, func_list in manifest.functions.items():
            if any(_name_matches(keyword_lower, fn) for fn in func_list):
                _add_file(result, mod)
                result.candidate_functions.append(keyword)
                result.decision = "MATCH"
                result.confidence = max(result.confidence, 0.8)
                result.reason = f"Function name match: {keyword} in {mod}"

    # Strategy 3: case-insensitive partial filename match
    if result.decision == "UNMATCHED":
        for keyword in _extract_keywords(claim_text):
            keyword_lower = keyword.lower()
            for f in manifest.files:
                fname = os.path.basename(f).lower()
                if keyword_lower in fname or fname.startswith(keyword_lower):
                    _add_file(result, f)
                    result.decision = "MATCH"
                    result.confidence = max(result.confidence, 0.7)
                    result.reason = f"Partial filename match: {f}"

    # Strategy 4: category file patterns
    if result.decision == "UNMATCHED":
        for pattern in _get_category_patterns(claim.category):
            for f in manifest.files:
                if pattern in f.lower():
                    _add_file(result, f)
                    result.decision = "MATCH"
                    result.confidence = max(result.confidence, 0.6)
                    result.reason = f"Category pattern match: {f}"

    # Strategy 5: hybrid retrieval (recall candidates for evidence grounding)
    # 查询带论文原句(claim_search_text):statement 常只有方法名,仓库代码几乎
    # 不与论文同名,必须靠"方法做什么"的语义才能召回真正的实现模块。
    # 类别模式命中(category pattern)是最弱的匹配(如 core_method 泛匹配
    # "model/net/layer"),不应阻断语义召回——否则候选只剩模式噪声文件。
    pattern_only = bool(result.reason and result.reason.startswith("Category pattern match"))
    hybrid_files: list[str] = []
    kw_files: list[str] = []
    if result.decision == "UNMATCHED" or result.confidence < 0.6 or pattern_only:
        hybrid_results = code_vector_store.hybrid_search(claim_search_text(claim), top_k=8)
        for hr in hybrid_results:
            _add_file(result, hr["file_path"])
            hybrid_files.append(hr["file_path"])
            if hr.get("class_name"):
                result.candidate_classes.append(hr["class_name"])
            if hr.get("function_name"):
                result.candidate_functions.append(hr["function_name"])
            if hr["score"] > 0.3:
                result.decision = "MATCH"
                result.confidence = max(result.confidence, min(hr["score"], 0.85))
                result.reason = f"Hybrid retrieval match (score={hr['score']}): {hr['file_path']}"
        # 符号/docstring 精确命中的文件(作者自述"此代码即论文概念")同列候选
        for kr in code_vector_store.keyword_search(claim_search_text(claim), top_k=6):
            if kr["file_path"] not in result.candidate_files:
                result.candidate_files.append(kr["file_path"])
            kw_files.append(kr["file_path"])

    # Deduplicate (keep first-seen order)
    result.candidate_files = _unique(result.candidate_files)
    result.candidate_classes = _unique(result.candidate_classes)
    result.candidate_functions = _unique(result.candidate_functions)

    # 模式命中在混合召回之前 append,会让 audit 阶段的 [:5] 截断把真正实现
    # 文件挤出候选(模式噪声占满名额)。精确命中与语义召回的文件排到模式命中前面。
    if pattern_only and (hybrid_files or kw_files):
        head = [f for f in (kw_files + hybrid_files) if f in result.candidate_files]
        tail = [f for f in result.candidate_files if f not in head]
        result.candidate_files = head + tail

    return result


def _add_file(result: MappingResult, file_path: str) -> None:
    if file_path and file_path not in result.candidate_files:
        result.candidate_files.append(file_path)


def _unique(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _name_matches(keyword_lower: str, func_name: str) -> bool:
    """Match dotted method names like ``Client.sendMessage`` against the
    keyword by either the full name or its bare component."""
    if keyword_lower == func_name.lower():
        return True
    bare = func_name.rsplit(".", 1)[-1].lower()
    return keyword_lower == bare


def _extract_keywords(text: str) -> list[str]:
    """Extract potential code keywords from claim text."""
    # Extract capitalized words that look like code names
    keywords = re.findall(r"[A-Z][a-zA-Z0-9_]+", text)
    # Also extract words after common patterns
    for pattern in [r"'(.*?)'", r"\"(.*?)\"", r"`(.*?)`"]:
        keywords.extend(re.findall(pattern, text))
    # 确定性顺序(见 code_vector_store._extract_keywords 注释)
    return sorted(set(keywords))[:10]


def _get_category_patterns(category: ClaimCategory) -> list[str]:
    """Get file patterns commonly associated with a claim category."""
    patterns = {
        ClaimCategory.CORE_METHOD: ["model", "net", "layer", "module", "block", "attention", "encoder", "decoder"],
        ClaimCategory.TRAINING: ["train", "loss", "optim", "scheduler", "dataset", "dataloader"],
        ClaimCategory.INFERENCE: ["infer", "predict", "demo", "test"],
        ClaimCategory.EVALUATION: ["eval", "metric", "benchmark", "score"],
        ClaimCategory.API_INTERFACE: ["api", "cli", "main", "config", "argument"],
        ClaimCategory.DATASET: ["dataset", "data", "dataloader"],
        ClaimCategory.CHECKPOINT: ["checkpoint", "weight", "pretrained", "model_zoo"],
        ClaimCategory.LICENSE: ["license", "licence"],
    }
    return patterns.get(category, [])
