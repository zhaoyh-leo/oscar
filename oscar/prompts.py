"""OSCAR 全部 LLM 提示词的唯一持有处(2026-09 集中迁移)。

不变量:本文件的提示词文本是 LLM 缓存键的组成部分 —— 缓存键
= sha256(messages+model+temperature+salt),任何一个字符的改动都会让
既有缓存全部失效,并破坏确定性验收(双跑字节一致)。因此:

- 含插值的正文以 builder 函数整块迁移(f-string/拼接表达式原样,
  含嵌套生成器、条件注入、{{}} 转义);
- 无插值的静态文本提为模块常量;
- 本文件只 import stdlib(typing),参数全为基本类型 → 零循环导入;
- 迁移 hunk 禁止跑任何格式化工具,直到双跑验证通过。

调用方负责:消息组装、temperature、降级重试、调试钩子等「契约」逻辑,
以及从结构化结果对象渲染文本(如 chunk 头/代码块包装)。
"""

from typing import Optional


def build_project_summary_prompt(manifest) -> str:
    """项目级摘要提示词(manifest 提供 files/classes 即可,duck-typing)。"""
    return f"""Summarize this open-source project in 2-3 sentences based on its code structure:

Repository files include:
{chr(10).join(manifest.files[:20])}

Key classes:
{chr(10).join([f"{k}: {v}" for k, v in list(manifest.classes.items())[:5]])}

Respond with JSON:
{{"project_summary": "2-3 sentence summary"}}
"""


def build_issue_analysis_prompt(
    issue: dict, comments: list[dict], relevant_claims: list[str]
) -> str:
    """GitHub issue 分析提示词。返回值由调用方解析为 (explanation, status)。"""
    return f"""Analyze the following GitHub issue and comments to determine if they explain why certain code components are missing from the repository.

Issue Title: {issue.get('title', '')}
Issue Body: {(issue.get('body', '') or '')[:1000]}

Comments:
{chr(10).join(f"- {c.get('author', 'unknown')}: {(c.get('body', '') or '')[:500]}" for c in comments[:5])}

Relevant Claims: {', '.join(relevant_claims)}

Does this issue explain why the claims' components are missing or incomplete? Possible implications:
- PLANNED: authors state the code will be released later.
- RESTRICTED: authors state the code cannot be shared (copyright/proprietary).
- MISSING: the issue confirms the component does not exist.
- UNCERTAIN: the issue is unrelated or inconclusive — use this for anything vague.

Provide your analysis with:
- explanation: a brief explanation of what the issue says (2-3 sentences, factual)
- status_implication: exactly one of "PLANNED", "RESTRICTED", "MISSING", "UNCERTAIN"
- relevant: true or false (false when the issue does not actually concern the claims)
"""


SYSTEM_PAPER_ASSISTANT = "You are a paper analysis assistant. Return only valid JSON."


def build_claim_extraction_prompt(window: str, readme_text: Optional[str] = None) -> str:
    """论文 claim 抽取提示词。readme 文本为空时 README 两行注入为空串。"""
    return f"""Extract the claims an open-source completeness audit should verify from the paper text below.

You are extracting what the paper *claims about its own contribution* — code or resources a released repository for this paper should contain. Work from the paper's own wording; include the method/component name when the paper gives one.

Categories and statement requirements (statement ≤ 200 chars, one compact sentence, always in the paper's own functional terms):
- core_method: an architecture / module / learning method the paper proposes. Statement = "<Name, if the paper names it>: what it does and how, in one clause" — e.g. "weighted-InfoNCE: weighted contrastive learning that weights cross-view training pairs by the IOU overlap of their ground areas". NEVER a bare name, NEVER the project/dataset name alone.
- Precedence: any method or component the paper itself proposes — architecture, module, loss, training objective, sampling strategy, or the framework configuration its method runs on (e.g. the weight-sharing Siamese ViT + MLP-head descriptor model) — belongs in core_method, even when the paper describes it in training terms or builds it from off-the-shelf backbones. Examples: "weighted-InfoNCE", "weight-sharing ViT descriptor", "mutually exclusive sampling".
- dataset / benchmark: data contributions. Statement = "<Name>: what it is (domain, scale, task framing)". A constructed dataset/benchmark belongs HERE, not in core_method.
- training: only training procedures that are not one of the paper's proposed method components (e.g. training schedule, pretraining protocol, hyperparameter practice).
- inference / evaluation: evaluation protocol, metrics, or localization task framing the paper introduces.
- implementation: engineering claims only (e.g. "code/weights will be released", reproducibility setup) — never for an architecture or model component.
- demo / release / api_interface / license: only when the paper makes an explicit statement about them.

Rules:
- Only include contributions the paper itself makes or adopts as its framework (e.g. "we use a pair of weight-sharing ViT models with an MLP head"). Do NOT mine related-work names.
- Every claim needs:
  - category: one of core_method, training, inference, evaluation, demo, benchmark, api_interface, dataset, checkpoint, license, implementation, release
  - statement: as specified above (function-first, ≤200 chars)
  - location: where it appears (Abstract, Introduction, "Method §3.x", ...)
  - snippet: the verbatim paper sentence(s) backing the claim (≤ 350 chars) — keep the original wording, this is used as search context later
- Extract as many distinct claims as the text supports; skip anything vague.

Paper text (claim-relevant window):
{window}

{"README:" if readme_text else ""}
{readme_text[:2000] if readme_text else ""}

Return JSON: a list of items with keys category, statement, location, snippet.
"""


GROUNDING_SYSTEM = (
    "You are a rigorous open-source completeness auditor. You are given one "
    "claim and a set of numbered code chunks retrieved from a repository. "
    "Determine whether the claim's functionality is implemented by the code shown.\n"
    "\n"
    "Rules:\n"
    "- Base conclusions ONLY on the code chunks provided. If the claim is not "
    "implemented by them, answer MISSING; if only part of the functionality "
    "exists, answer INCOMPLETE.\n"
    "- A stub, TODO, bare 'pass' body, or NotImplementedError is NOT an "
    "implementation.\n"
    "- Code identifiers and file names often differ from the paper's naming: "
    "the same component ships under generic or different names (e.g. the "
    "paper's method implemented in a file called loss.py). Judge by whether "
    "the shown code implements the described functionality — never answer "
    "MISSING merely because no symbol matches the claim's name.\n"
    "- A pretrained encoder used as a descriptor extractor is routinely "
    "instantiated without its final classification head (e.g. timm "
    "num_classes=0); the pooled encoder features ARE the descriptor, and "
    "the classification head (which outputs pretraining-task logits) is not "
    "part of the descriptor. Do not treat its absence as a missing "
    "implementation unless the claim explicitly requires that head's "
    "outputs.\n"
    "- If you answer MISSING, briefly state what the shown chunks do instead, "
    "so the reader sees the code you considered.\n"
    "- For every relevant chunk, explain in natural language what that code "
    "actually does — a concise code summary (what_it_does), never a restatement "
    "of the file name.\n"
    "- locations must reference the provided chunk refs (C1, C2, ...). Never "
    "invent files, symbols, or line numbers that were not shown.\n"
    "- snippet must quote code verbatim from the referenced chunk."
)


def build_grounding_user_prompt(
    *,
    project_name: str,
    repo_url: str,
    claim_id: str,
    category: str,
    statement: str,
    status: str,
    confidence: float,
    paper_context: str = "",
    chunks_text: list[str],
    max_locations: int,
) -> str:
    """证据接地 user 提示词。chunks_text 为调用方渲染好的代码块字符串列表。

    拼接顺序即原实现:head →(paper_context 非空时追加一行)→ tail。
    """
    user_head = (
        f"Project: {project_name}\n"
        f"Repository: {repo_url}\n"
        f"Claim ID: {claim_id} | Category: {category}\n"
        f"Claim: {statement}\n"
    )
    if paper_context:
        user_head += f"Paper context (verbatim source sentence of the claim): {paper_context}\n"
    user_tail = (
        f"Provisional audit status: {status} "
        f"(confidence {confidence:.2f}) — treat as provisional; verify "
        f"against the code below.\n\n"
        f"Code chunks:\n\n{chr(10).join(chunks_text)}\n\n"
        "Answer with the structured result: an overall verdict "
        "(VERIFIED / INCOMPLETE / MISSING / UNCERTAIN), a confidence in [0,1], "
        f"an explanation of 2-4 sentences, and up to {max_locations} locations "
        "(each referencing a chunk ref above, with what_it_does and a verbatim "
        "snippet)."
    )
    return user_head + user_tail


GROUNDING_DEGRADE_INSTRUCTION = (
    "Return raw JSON only, matching GroundingVerdict: "
    '{"verdict": "...", "confidence": 0.0, "explanation": "...", '
    '"locations": [{"chunk_ref": "C1", "what_it_does": "...", '
    '"snippet": "..."}]}'
)
