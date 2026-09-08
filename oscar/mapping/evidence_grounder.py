"""Evidence grounding — LLM reads the actual code and grounds findings.

背景问题:全项目旧版 LLM 调用无一收到代码原文,judge 只用
evidence_summary[:300](文件名列表)重写 explanation,证据永远停在文件级、
内容像关键词匹配而非理解代码。

本模块取代 fusion/judge,对每条「可代码验证」的 finding:
1. hybrid_search 检索 top-k 代码块(函数/类体原文,带行号)——
   纯本地检索,无失败风险;
2. 把 claim + 暂态状态 + 代码原文喂给 LLM(structured output),
   输出整体 verdict + 每条相关代码的 what_it_does 自然语言摘要;
3. 位置信息以 chunk 记录为准(chunk_ref 锚定,不信 LLM 自报坐标),
   回写 EvidenceDetail(file/line/symbol/snippet/code_explanation)
   与 explanation;
4. 检索无命中 → 规则化 MISSING,不调 LLM 去「确认不存在」
   (幻觉高发区);LLM 失败 → 保留原证据、explanation 留痕标注、
   confidence 封顶 —— 绝不静默吞错;
5. 大块兜底:初裁非 VERIFIED 时,把初窗文件全文再送 LLM 复核 ——
   小块只露函数/类体,同文件散落(其他函数/模块级)的实现面是误报
   INCOMPLETE/MISSING 的主源(小块优先、大块兜底,见 _whole_file_reground)。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Optional

from oscar.config import config
from oscar.llm.client import llm_client
from oscar.prompts import (
    GROUNDING_DEGRADE_INSTRUCTION,
    GROUNDING_SYSTEM,
    build_grounding_user_prompt,
    build_whole_file_grounding_prompt,
)
from oscar.models.schemas import (
    AuditFinding, AuditState, ClaimStatus, EvidenceDetail, GroundingVerdict,
)
from oscar.mapping.code_vector_store import code_vector_store
from oscar.paper.analyzer import claim_search_text
from oscar.report.category_policy import (
    CODE_GROUNDED_CATEGORIES, category_search_query,
)

logger = logging.getLogger(__name__)

# 检索/接地参数绑定 config.yaml retrieval 段(默认 = 原硬编码现值,见
# oscar/config.py::RetrievalConfig —— 25 处下游引用经同名常量自动跟随)。
_TOP_K = config.retrieval.top_k
# hybrid 检索送入 LLM 的候选代码块数(长论文原文查询会把 BM25 的真命中压到
# 6 名之后;窗口开宽让实现块进入 LLM 视野)
_MAX_CHUNKS = config.retrieval.max_chunks
# hybrid + mapper 结构化召回合计上限
_MAX_CHUNK_CHARS = config.retrieval.max_chunk_chars
# 每块代码截断长度(大类的完整类体可数千字符,截太短会只露出头部、
# 藏住真实实现语义)
_MAX_LOCATIONS = config.retrieval.max_locations
# 报告 evidence details 的定位上限
_CONF_CAP = config.retrieval.conf_cap
# LLM 置信度上限(防过度自信)
_DEGRADE_CONF_CAP = config.retrieval.degrade_conf_cap
# LLM 不可用时的置信度封顶
_RARE_SLOTS = config.retrieval.rare_slots
# 稀有内容词扫描专用槽位数(见 _assemble_candidate_chunks)
_MAPPER_MIN = config.retrieval.mapper_min
# mapper 候选文件块保底(与论文不同名的实现仅经此可达)
# 大块兜底预算(小块优先、只在初裁「不完整」时整读文件复核;验证有效后再
# 提升为 config.yaml retrieval 段可调项)
_WHOLE_FILE_MAX_FILES = 3        # 整读文件数上限(初窗文件 + mapper 候选)
_WHOLE_FILE_TOTAL_CHARS = 60000  # 全文合计字符预算
_WHOLE_FILE_SKIP_CHARS = 40000   # 单文件超此长度不整读(其类级块初窗已覆盖)
_VERDICT_STATUS = {
    "VERIFIED": ClaimStatus.VERIFIED,
    "INCOMPLETE": ClaimStatus.INCOMPLETE,
    "MISSING": ClaimStatus.MISSING,
    "UNCERTAIN": ClaimStatus.UNCERTAIN,
}


def ground_finding(
    finding: AuditFinding,
    state: AuditState,
) -> tuple[Optional[AuditFinding], Optional[str]]:
    """Ground one finding on actual repository code.

    Returns ``(updated_finding | None, error_note | None)``. Returning None
    for the finding means "no change needed" (out of scope category, or
    issue-level RESTRICTED/PLANNED evidence is authoritative).
    """
    if finding.category not in CODE_GROUNDED_CATEGORIES:
        return None, None
    if finding.status in (ClaimStatus.RESTRICTED, ClaimStatus.PLANNED):
        return None, None  # issue 官方陈述是第一档证据,不送 LLM 改写
    if not code_vector_store.chunks:
        return None, None  # 无可检索代码(如非 Python 仓库)→ 不做裁决

    manifest = state.repository_manifest

    # 检索查询与 paper context 都带论文原句(claim.source.content):
    # 仓库代码几乎不与论文同名,靠 claim 的"方法做什么"语义召回真正实现。
    claim = next(
        (c for c in (state.claims or []) if c.claim_id == finding.claim_id), None
    )
    search_text = claim_search_text(claim) if claim is not None else finding.statement
    paper_context = ""
    if claim is not None and claim.source and claim.source.content:
        loc = claim.source.location or ""
        paper_context = f"[{loc}] {claim.source.content}" if loc else claim.source.content

    # 查询:claim 带论文原句时直接用它(类别提示词如 "model network module
    # layer forward" 会稀释方法语义、把第三方 matcher/backbone 噪声顶上排名);
    # 只有 statement 无 paper context 的模块级 finding 才补类别提示词。
    if claim is not None and claim.source and claim.source.type == "paper" and claim.source.content:
        query = search_text
    else:
        query = category_search_query(search_text, finding.category)
    results = code_vector_store.hybrid_search(query, top_k=_TOP_K)

    if not results:
        # 检索无任何代码块:规则化 MISSING,不调 LLM 确认不存在
        new = finding.model_copy(deep=True)
        new.status = ClaimStatus.MISSING
        new.confidence = 0.3
        new.explanation = (
            "No code chunks could be retrieved to ground this claim, so no "
            "repository code was examined. Status set to MISSING by rule."
        )
        return new, None

    # 结构化召回兜底:检索可能漏掉实现模块(代码不与论文同名)。audit 阶段
    # mapper 已把「精确/部分文件名、符号名、类别模式」命中的文件写进
    # evidence_details —— 把它们的类级代码块并入上下文,让 LLM 看到真正
    # 的实现,而不只是检索 top-k。
    # 组装顺序(总块数受 _MAX_CHUNKS 封顶):
    #   1. keyword 精确命中置顶(符号/docstring 词,作者自述"此代码即论文
    #      概念")取前 6 —— 最强信号;
    #   2. hybrid 检索结果(去重追补);
    #   3. 稀有内容词扫描腿:claim 与实现共享的低频短词(3-4 字母,进不了
    #      content-match)专用 _RARE_SLOTS 槽,从 hybrid 尾部(弱 RRF 噪声)
    #      裁出位置——这类实现代码只能经此可达;
    #   4. mapper 候选文件的类级代码块(至少 _MAPPER_MIN 块,与论文不同名
    #      的实现只经此路径)。
    ordered = _assemble_candidate_chunks(
        query, results, _mapper_candidate_files(finding)
    )

    chunk_map = {f"C{i + 1}": r for i, r in enumerate(ordered)}
    project_name = state.project.get("name", "")
    repo_url = state.project.get("repository_url", "")

    try:
        verdict = _call_grounding_llm(finding, chunk_map, project_name, repo_url, paper_context)
        if verdict is None:
            return _degrade(finding, "LLM grounding returned no usable result")
    except Exception as exc:  # noqa: BLE001 — 降级留痕,不阻断审计
        logger.warning("Grounding failed for %s: %s", finding.claim_id, exc)
        return _degrade(finding, f"{type(exc).__name__}: {exc}")

    status = _VERDICT_STATUS.get(verdict.verdict)
    if status is None:
        return _degrade(finding, f"unrecognized verdict {verdict.verdict!r}")

    # Resolve LLM locations onto authoritative chunk records (chunk_ref anchor)
    grounded: list[EvidenceDetail] = []
    seen_keys: set[tuple] = set()
    for loc in verdict.locations or []:
        if len(grounded) >= _MAX_LOCATIONS:
            break
        ref = (loc.chunk_ref or "").strip().upper()
        chunk = chunk_map.get(ref)
        if chunk is None:
            continue
        content = chunk.get("content") or ""
        snippet = (loc.snippet or "").strip()
        if not snippet or snippet not in content:
            snippet = _first_significant_line(content)
        cls = chunk.get("class_name") or None
        fn = chunk.get("function_name") or None
        key = (chunk["file_path"], chunk.get("line_start"), cls, fn)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        grounded.append(EvidenceDetail(
            file_path=chunk["file_path"],
            line_number=chunk.get("line_start", 0),
            line_end=chunk.get("line_end", 0),
            class_name=cls,
            function_name=fn,
            snippet=snippet[:240],
            label=_symbol_label(cls, fn, chunk["file_path"]),
            code_explanation=(loc.what_it_does or "").strip(),
        ))

    # 裁决一致性守卫:VERIFIED/INCOMPLETE 必须能指出具体代码
    if status in (ClaimStatus.VERIFIED, ClaimStatus.INCOMPLETE) and not grounded:
        status = ClaimStatus.UNCERTAIN
        verdict.explanation = (
            (verdict.explanation or "")
            + " The verdict could not be anchored to any of the provided code "
            "chunks, so it is reported as UNCERTAIN."
        ).strip()

    # 大块兜底(小块优先、只在「实现不完整」时补):初裁非 VERIFIED → 整读
    # 初窗文件全文复核。小块只露函数/类体,同文件其他函数/模块级实现面
    # 从不入窗,是误报 INCOMPLETE/MISSING 的主源。复核成功以复核结果为准
    # (重新裁决 + 全文内精确定位);失败/无文件可读 → 维持初裁。
    revised = None
    if status in (ClaimStatus.MISSING, ClaimStatus.INCOMPLETE, ClaimStatus.UNCERTAIN):
        revised = _whole_file_reground(
            finding, chunk_map, state, verdict, paper_context
        )
        if revised is not None:
            status, grounded, verdict = revised
    # 复核裁决同样必须锚得住(整文件上下文亦不豁免一致性守卫)
    if (
        revised is not None
        and status in (ClaimStatus.VERIFIED, ClaimStatus.INCOMPLETE)
        and not grounded
    ):
        status = ClaimStatus.UNCERTAIN
        verdict.explanation = (
            (verdict.explanation or "")
            + " The verdict could not be anchored to the provided complete "
            "files, so it is reported as UNCERTAIN."
        ).strip()

    new = finding.model_copy(deep=True)
    new.status = status
    new.confidence = min(max(float(verdict.confidence), 0.05), _CONF_CAP)

    # 保留模块自身带真实位置的行级证据(行号/片段),丢弃纯文件名噪声行
    kept = [
        d for d in finding.evidence_details
        if (d.line_number > 0 or d.snippet or d.class_name or d.function_name)
    ]
    new.evidence_details = _dedupe_details(kept + grounded)

    if status == ClaimStatus.MISSING:
        new.evidence_summary = "No repository code implements this claim"
    elif grounded:
        locs = [
            f"{d.file_path}#L{d.line_number}"
            + (f" {d.class_name}.{d.function_name}" if d.class_name and d.function_name
               else f" {d.function_name}" if d.function_name
               else f" class {d.class_name}" if d.class_name else "")
            for d in grounded
        ]
        new.evidence_summary = ", ".join(locs[:5])

    explanation = (verdict.explanation or "").strip()
    if not explanation:
        explanation = (
            f"The claim '{finding.statement}' was judged {status.value} based on "
            "the repository code provided to grounding."
        )
    new.explanation = explanation
    return new, None


def _mapper_candidate_files(finding: AuditFinding) -> list[str]:
    """Files the mapper flagged for this finding (symbol/file/pattern match).

    audit 阶段 mapper 的命中以「candidate file」标签写入 evidence_details
    (纯文件行:无行号/符号/片段)。模块自带的行级证据行不在此列 —— 它们带
    行号/符号/片段,已被既有代码判据排除,不会误充候选。
    """
    out: list[str] = []
    for d in finding.evidence_details or []:
        if not d.file_path:
            continue
        is_candidate_row = d.label == "candidate file" or (
            not d.class_name
            and not d.function_name
            and not d.snippet
            and not d.line_number
        )
        if is_candidate_row and d.file_path not in out:
            out.append(d.file_path)
    return out


def _assemble_candidate_chunks(
    query: str,
    results: list[dict],
    candidate_files: list[str],
) -> list[dict]:
    """Assemble the ≤ _MAX_CHUNKS code-chunk window shown to the grounding LLM.

    纯函数(只读 store),离线探针可直接复用。槽位纪律:
    - kw 置顶 ≤6(最强信号,永不裁剪);
    - hybrid 去重追补,超窗时从尾部(弱 RRF/弱语义行)裁;
    - 稀有内容词扫描(rare_token_search)命中时保证 _RARE_SLOTS 槽 —— 这些
      块是语义与符号检索都够不到的代码(如只有低频短标识符可匹配的实现),
      从 hybrid 尾部裁出位置而不是挤掉 mapper 保底;
    - mapper 候选文件块保底 ≥ _MAPPER_MIN(非同名实现入口)。
    """
    kw_picks = code_vector_store.keyword_search(query, top_k=6)
    have = {r["chunk_id"] for r in kw_picks}
    ordered = list(kw_picks)
    for r in results:
        if r["chunk_id"] in have:
            continue
        have.add(r["chunk_id"])
        ordered.append(r)

    # rare-token 腿:只取未进窗的候选
    rare = [
        r for r in code_vector_store.rare_token_search(query, top_k=_RARE_SLOTS)
        if r["chunk_id"] not in have
    ]
    if rare:
        # 需要为 rare 块与 mapper 保底腾位时,裁 hybrid 尾(绝不裁 kw 置顶)
        over = len(ordered) + len(rare) + _MAPPER_MIN - _MAX_CHUNKS
        if over > 0 and candidate_files:
            cut_at = max(len(kw_picks), len(ordered) - over)
            ordered = ordered[:cut_at]
        for r in rare:
            have.add(r["chunk_id"])
            ordered.append(r)

    if len(ordered) < _MAX_CHUNKS and candidate_files:
        extras = code_vector_store.chunks_for_files(
            candidate_files,
            budget=max(_MAPPER_MIN, _MAX_CHUNKS - len(ordered)),
        )
        for e in extras:
            if e["chunk_id"] not in have:
                have.add(e["chunk_id"])
                ordered.append(e)
    return ordered[: _MAX_CHUNKS]


def _degrade(finding: AuditFinding, reason: str) -> tuple[AuditFinding, str]:
    """LLM unavailable/malformed: keep evidence, mark the report, cap confidence."""
    note = f"[code grounding unavailable for {finding.claim_id}: {reason}]"
    new = finding.model_copy(deep=True)
    new.confidence = min(new.confidence, _DEGRADE_CONF_CAP)
    if new.explanation:
        new.explanation = f"{new.explanation}\n\n{note}"
    else:
        new.explanation = note
    return new, note


def _call_grounding_llm(
    finding: AuditFinding,
    chunk_map: dict[str, dict],
    project_name: str,
    repo_url: str,
    paper_context: str = "",
) -> Optional[GroundingVerdict]:
    """One structured LLM call with the actual code chunks as context."""
    chunks_text = []
    for ref, r in chunk_map.items():
        content = (r.get("content") or "")[:_MAX_CHUNK_CHARS]
        header_parts = [f"[{ref}] {r['file_path']}", f"lines {r.get('line_start', 0)}-{r.get('line_end', 0)}"]
        if r.get("class_name"):
            header_parts.append(f"class {r['class_name']}")
        if r.get("function_name"):
            header_parts.append(f"function {r['function_name']}")
        chunks_text.append(f"{' | '.join(header_parts)}\n```python\n{content}\n```")

    user = build_grounding_user_prompt(
        project_name=project_name,
        repo_url=repo_url,
        claim_id=finding.claim_id,
        category=finding.category.value,
        statement=finding.statement,
        status=finding.status.value,
        confidence=finding.confidence,
        paper_context=paper_context,
        chunks_text=chunks_text,
        max_locations=_MAX_LOCATIONS,
    )

    messages = [
        {"role": "system", "content": GROUNDING_SYSTEM},
        {"role": "user", "content": user},
    ]
    # 确定性调试钩子:OSCAR_DEBUG_GROUNDING 时记录每次 grounding 调用的
    # prompt 指纹(claim_id + messages sha + 各 C 槽定位),供跨 run diff 定位
    # 缓存未命中来源。只读不改,无该 env 时零开销。
    if os.environ.get("OSCAR_DEBUG_GROUNDING"):
        try:
            parts = [f"{finding.claim_id}\tsha={hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()[:16]}"]
            for ref, r in chunk_map.items():
                tail = (r.get("file_path") or "").replace("\\", "/").rsplit("/", 1)[-1]
                parts.append(f"{ref}:{tail}:{r.get('line_start')}-{r.get('line_end')}"
                             f":{r.get('class_name') or ''}.{r.get('function_name') or ''}")
            with open("grounding_debug.log", "a", encoding="utf-8") as _f:
                _f.write("\t".join(parts) + "\n")
        except Exception:
            pass
    return _structured_grounding_call(messages)


def _structured_grounding_call(messages: list[dict]) -> Optional[GroundingVerdict]:
    """One structured grounding call (shared by chunk pass and whole-file pass),
    with the raw-JSON degrade fallback when function calling is unavailable."""
    try:
        result = llm_client.structured_output(GroundingVerdict, messages, temperature=0.1)
        if isinstance(result, GroundingVerdict):
            return result
        return None
    except Exception:
        # 兜底:模型不支持 function calling 时退回 chat_json 手写 JSON
        try:
            raw = llm_client.chat_json(
                messages + [{"role": "user", "content": GROUNDING_DEGRADE_INSTRUCTION}],
                temperature=0.1,
            )
            if isinstance(raw, dict):
                return GroundingVerdict.model_validate(raw)
        except Exception:
            pass
        return None


def _whole_file_reground(
    finding: AuditFinding,
    chunk_map: dict[str, dict],
    state: AuditState,
    prior_verdict: GroundingVerdict,
    paper_context: str = "",
) -> Optional[tuple[ClaimStatus, list[EvidenceDetail], GroundingVerdict]]:
    """整文件复核(大块兜底)。

    文件集 = 初窗 chunk 所在文件(窗口序)+ mapper 候选文件,确定性去重、
    按预算整读;LLM 按全文复核初裁;锚定行号由 snippet 在文件全文中的逐字
    精确匹配反推(非 LLM 自报坐标,锚不上即弃 —— 宁缺毋假)。返回
    ``(status, grounded_rows, verdict)``;无文件可读或复核调用失败返回 None,
    调用方维持初裁。
    """
    files: list[str] = []
    seen: set[str] = set()
    for r in chunk_map.values():
        fp = (r.get("file_path") or "").strip()
        if fp and fp not in seen:
            seen.add(fp)
            files.append(fp)
    for fp in _mapper_candidate_files(finding):
        if fp and fp not in seen:
            seen.add(fp)
            files.append(fp)
    if not files:
        return None

    repo_path = state.project.get("clone_path", "")
    texts: list[tuple[str, str]] = []  # (manifest 风格路径, 全文)
    total = 0
    for fp in files:
        if len(texts) >= _WHOLE_FILE_MAX_FILES:
            break
        if total >= _WHOLE_FILE_TOTAL_CHARS:
            break
        open_path = os.path.join(repo_path, fp.replace("\\", "/")) if repo_path else fp
        try:
            with open(open_path, encoding="utf-8") as fh:  # 文本模式 → \r\n 归一
                content = fh.read()
        except (OSError, UnicodeError):
            continue
        if len(content) > _WHOLE_FILE_SKIP_CHARS:
            continue  # 超大文件不整读(其类级块已入初窗);预算留给小文件
        if total + len(content) > _WHOLE_FILE_TOTAL_CHARS and texts:
            break  # 放不下就不再塞半截文件
        total += len(content)
        texts.append((fp, content))
    if not texts:
        return None

    files_text = [
        f"[F{i + 1}] {fp} ({content.count(chr(10)) + 1} lines)\n"
        f"```python\n{content}\n```"
        for i, (fp, content) in enumerate(texts)
    ]
    user = build_whole_file_grounding_prompt(
        project_name=state.project.get("name", ""),
        repo_url=state.project.get("repository_url", ""),
        claim_id=finding.claim_id,
        category=finding.category.value,
        statement=finding.statement,
        paper_context=paper_context,
        prior_status=prior_verdict.verdict,
        prior_confidence=float(prior_verdict.confidence),
        files_text=files_text,
        max_locations=_MAX_LOCATIONS,
    )
    verdict = _structured_grounding_call(
        [
            {"role": "system", "content": GROUNDING_SYSTEM},
            {"role": "user", "content": user},
        ]
    )
    if verdict is None:
        return None
    status = _VERDICT_STATUS.get(verdict.verdict)
    if status is None:
        return None

    grounded: list[EvidenceDetail] = []
    seen_keys: set[tuple] = set()
    for loc in verdict.locations or []:
        if len(grounded) >= _MAX_LOCATIONS:
            break
        ref = (loc.chunk_ref or "").strip().upper()
        if not ref.startswith("F") or not ref[1:].isdigit():
            continue
        idx = int(ref[1:]) - 1
        if idx < 0 or idx >= len(texts):
            continue
        fp, content = texts[idx]
        snippet = (loc.snippet or "").strip()
        if len(snippet) < 8:
            continue
        pos = content.find(snippet)
        if pos < 0:
            continue  # 非逐字 → 弃,绝不用模糊坐标
        line = content.count("\n", 0, pos) + 1
        key = (fp, line)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        grounded.append(EvidenceDetail(
            file_path=fp,
            line_number=line,
            line_end=line + snippet.count("\n"),
            snippet=snippet[:240],
            label=_symbol_label(None, None, fp),
            code_explanation=(loc.what_it_does or "").strip(),
        ))
    return status, grounded, verdict


def _symbol_label(class_name: Optional[str], function_name: Optional[str], file_path: str) -> str:
    """Human label for an evidence row: symbol when known, else file base."""
    if class_name and function_name:
        return f"{class_name}.{function_name}()"
    if function_name:
        return f"{function_name}()"
    if class_name:
        return f"class {class_name}"
    base = file_path.rsplit("/", 1)[-1]
    return base


def _first_significant_line(content: str) -> str:
    for line in content.split("\n"):
        if line.strip():
            return line.strip()[:200]
    return content[:200]


def _dedupe_details(details: list[EvidenceDetail]) -> list[EvidenceDetail]:
    """Drop duplicate rows on (file, line, class, function) — modules may
    already report a precise row that grounding re-adds."""
    out: list[EvidenceDetail] = []
    seen: set[tuple] = set()
    for d in details:
        key = (d.file_path, d.line_number, d.class_name, d.function_name)
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out
