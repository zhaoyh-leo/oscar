"""LangGraph workflow for the OSCAR audit pipeline.

与旧版（ThreadPoolExecutor 内部并行）相比，本实现：

1. 真正的并行审计：
   - 6 个审计维度是独立的 LangGraph Node，调度器完全感知并行状态。
   - 从 Planner 节点 Fan-out 到 6 个审计 Node，利用 LangGraph 原生并发。
   - 通过 State 的 Reducer (operator.add) 自动合并 findings，无需手动管理线程池。

2. 解耦的进度反馈（Custom Events 机制）：
   - 每个 Node 通过 adispatch_custom_event 显式派发 on_node_start / on_node_end。
   - AuditCallbackHandler 通过 on_custom_event 精确消费，渲染交由外部 ProgressHandler。
   - 核心审计代码零 UI 依赖。
   - 注意：不使用 on_chain_start —— 它是运行级回调，编译图根运行 serialized 为
     None，且节点内部的 LLM 调用也会触发，无法用于识别节点生命周期。

3. 无脚本思维：
   - 每个 Node 是纯函数，无副作用，仅依赖 AuditState 输入。
   - 并行节点之间不共享可变状态，LangGraph 的 State 管理保证并发安全。

Graph flow:
  START → Repository Loader → Paper Resolver → Paper Analyzer →
  Repository Analyzer → Planner ─┬─ Core Methods ─┐
                                  ├─ Training ─────┤
                                  ├─ Inference ────┤
                                  ├─ API ──────────┤→ Issue Investigator → ...
                                  ├─ Resources ────┤
                                  └─ License ──────┘
"""

from __future__ import annotations

import asyncio
import inspect
import os
import shutil
from typing import Any, Callable, Optional

from langgraph.graph import StateGraph, END
from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.runnables.config import RunnableConfig

from oscar.config import config
from oscar.llm.client import llm_client
from oscar.prompts import build_project_summary_prompt
from oscar.models.schemas import (
    AuditFinding, AuditState, CodeSummary, Evidence, EvidenceType,
)
from oscar.repository.loader import clone_repository, get_readme_content
from oscar.repository.analyzer import generate_manifest
from oscar.repository.stub_detector import detect_stubs
from oscar.paper.resolver import resolve_paper, chunk_paper
from oscar.paper.analyzer import extract_claims_from_paper
from oscar.mapping.code_vector_store import code_vector_store
from oscar.audit.core_method import audit_core_methods
from oscar.audit.training import audit_training
from oscar.audit.inference_eval import audit_inference_eval
from oscar.audit.api_interface import audit_api_interface
from oscar.audit.resources import audit_resources
from oscar.audit.license_audit import audit_license
from oscar.audit.issues_pr import investigate_issues
from oscar.fusion.aggregator import aggregate_evidence
from oscar.mapping.evidence_grounder import ground_finding
from oscar.report.generator import generate_report, save_markdown_report, save_json_report
from oscar.utils.progress_handler import (
    AuditCallbackHandler, BaseProgressHandler, NullProgressHandler,
)
from oscar.mapping.claim_code_mapper import map_claim_to_code


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _get_project_name(repo_url: str) -> str:
    """Extract project name from GitHub URL."""
    parts = repo_url.strip("/").split("/")
    return parts[-1].replace(".git", "") if parts else "unknown"


def _get_output_dir(project_name: str) -> str:
    """Get project-specific output directory."""
    out_dir = config.paths.output_dir / project_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir)


# 人类可读的节点名（用于进度条展示）
_NODE_LABELS: dict[str, str] = {
    "repository_loader": "Repository Loader",
    "paper_resolver": "Paper Resolver",
    "paper_analyzer": "Paper Analyzer",
    "repository_analyzer": "Repository Analyzer",
    "planner": "Planner",
    "audit_core_methods": "Core Methods",
    "audit_training": "Training",
    "audit_inference": "Inference & Eval",
    "audit_api": "API Interface",
    "audit_resources": "Resources",
    "audit_license": "License",
    "issue_investigator": "Issue Investigator",
    "evidence_aggregator": "Evidence Aggregator",
    "evidence_grounder": "Evidence Grounding",
    "report_generator": "Report Generator",
    "cleanup": "Cleanup",
}


def _with_progress(node_id: str) -> Callable:
    """将节点包装为异步函数，并在执行前后显式派发 Custom Event。

    为什么必须由节点显式派发，而不是在全局回调里监听 on_chain_start：
    - on_chain_start 属于运行级回调，LangGraph 的编译图根、节点内部每次 LLM
      调用都会触发它；图根的 serialized 按设计为 None，无法据此识别“哪个节点”。
    - 由节点自己通过 adispatch_custom_event 派发事件，事件源与业务一一对应，
      handler 通过 on_custom_event 精确接收，无需任何猜测或过滤。
    """
    label = _NODE_LABELS.get(node_id, node_id)

    def decorator(fn: Callable):
        async def wrapper(state: AuditState, config: RunnableConfig) -> dict[str, Any]:
            await adispatch_custom_event(
                "on_node_start", {"node_name": label}, config=config
            )
            try:
                result = fn(state)
                if inspect.isawaitable(result):
                    result = await result
                return result
            finally:
                await adispatch_custom_event(
                    "on_node_end", {"node_name": label}, config=config
                )
        # 注意：不能使用 functools.wraps —— 它会使 inspect.signature 跟随
        # __wrapped__ 返回原函数 (state) 的签名，导致 LangGraph 不传 config。
        wrapper.__name__ = f"node_{node_id}"
        return wrapper

    return decorator


def _generate_code_summary(repo_path: str, manifest) -> CodeSummary:
    """Generate a multi-level code summary using LLM for context."""
    summary = CodeSummary()

    # Project-level summary from key files
    top_files = []
    for f in manifest.files[:30]:
        content = ""
        ext = os.path.splitext(f)[1].lower()
        if ext == ".py":
            try:
                with open(os.path.join(repo_path, f), "r") as fh:
                    content = fh.read()[:500]
            except Exception:
                pass
        if content:
            top_files.append(f"File: {f}\n{content[:300]}")

    prompt = build_project_summary_prompt(manifest)
    try:
        result = llm_client.chat_json([{"role": "user", "content": prompt}], temperature=0.1)
        summary.project_summary = result.get("project_summary", "")
    except Exception:
        pass

    # File-level summaries for key files
    for mod in manifest.python_modules[:10]:
        content = ""
        try:
            with open(os.path.join(repo_path, mod), "r") as fh:
                content = fh.read()[:200]
        except Exception:
            pass
        if content:
            summary.file_summaries[mod] = content[:200].replace("\n", " ")[:100]

    # Class/method-level summaries
    for mod, class_list in manifest.classes.items():
        for cls in class_list:
            summary.class_method_summaries[f"{mod}:{cls}"] = f"class {cls} in {mod}"
    for mod, func_list in manifest.functions.items():
        for func in func_list[:5]:
            summary.class_method_summaries[f"{mod}:{func}"] = f"function {func} in {mod}"

    return summary


# ---------------------------------------------------------------------------
# Node Functions
# ---------------------------------------------------------------------------

@_with_progress("repository_loader")
def node_repository_loader(state: AuditState) -> dict[str, Any]:
    """Clone repository and prepare for analysis."""
    repo_url = state.project.get("repository_url", "")
    if not repo_url:
        return {"errors": ["No repository URL provided"]}

    try:
        project_name = _get_project_name(repo_url)
        clone_path = clone_repository(repo_url)

        return {
            "project": {
                "name": project_name,
                "repository_url": repo_url,
                "clone_path": clone_path,
            }
        }
    except Exception as e:
        return {"errors": [f"Repository loader error: {str(e)}"]}


@_with_progress("paper_resolver")
def node_paper_resolver(state: AuditState) -> dict[str, Any]:
    """Resolve paper information."""
    repo_url = state.project.get("repository_url", "")
    clone_path = state.project.get("clone_path", "")
    paper_url = state.project.get("paper_url", "")

    paper = resolve_paper(repo_url, paper_url=paper_url or None)
    if not paper and clone_path:
        readme = get_readme_content(clone_path)
        if readme:
            import re
            arxiv_match = re.search(r"arxiv\.org/(?:abs|pdf)/(\d+\.\d+)", readme)
            if arxiv_match:
                paper = resolve_paper(repo_url, paper_url=f"https://arxiv.org/abs/{arxiv_match.group(1)}")

    return {"paper": paper}


@_with_progress("paper_analyzer")
def node_paper_analyzer(state: AuditState) -> dict[str, Any]:
    """Analyze paper and README to extract claims, then chunk for Map-Reduce."""
    clone_path = state.project.get("clone_path", "")
    readme = get_readme_content(clone_path) if clone_path else ""

    claims = extract_claims_from_paper(state.paper, readme)

    # Chunk paper for Map-Reduce processing
    if state.paper:
        chunk_paper(state.paper)
        # Index paper chunks into vector store for semantic retrieval
        from oscar.paper.vector_store import paper_vector_store
        paper_vector_store.index_paper(state.paper)
        # Persist to disk
        if state.paper.arxiv_id:
            paper_vector_store.save(state.paper.arxiv_id)

    return {"claims": claims, "readme_content": readme}


@_with_progress("repository_analyzer")
def node_repository_analyzer(state: AuditState) -> dict[str, Any]:
    """Analyze repository structure and generate manifest."""
    clone_path = state.project.get("clone_path", "")
    if not clone_path or not os.path.isdir(clone_path):
        return {"errors": ["Repository not cloned"]}

    manifest = generate_manifest(clone_path)

    # Detect stubs
    stub_details = detect_stubs(clone_path)
    manifest.stub_files = list(stub_details.keys())
    manifest.stub_details = stub_details

    # Index code into vector store for hybrid retrieval (repo_url 作持久化键)
    repo_url = state.project.get("repository_url", "")
    code_vector_store.index_repository(manifest, clone_path, repo_url=repo_url)

    # Generate multi-level code summary via LLM
    if manifest.python_modules:
        try:
            summary = _generate_code_summary(clone_path, manifest)
            manifest.code_summary = summary
        except Exception:
            pass

    return {"repository_manifest": manifest}


@_with_progress("planner")
def node_planner(state: AuditState) -> dict[str, Any]:
    """Prepare claims and map them to code before fan-out to audit nodes.

    与旧版 chunk_processor 相比：
    - 不再在线程池内运行审计，只负责将 claim 映射到代码。
    - 映射结果存储在 state.mapping_results 中，供后续审计节点使用。
    - 所有 claims 都在 state.claims 中，审计节点按 category 自行过滤。
    """
    manifest = state.repository_manifest
    repo_path = state.project.get("clone_path", "")
    mapping_results: dict[str, Any] = {}

    if not manifest:
        return {"mapping_results": mapping_results}

    # Map each claim to code for later use by audit nodes
    for claim in state.claims:
        try:
            mapping = map_claim_to_code(claim, manifest, repo_path)
            mapping_results[claim.claim_id] = mapping
        except Exception:
            pass

        # If mapping found, add evidence
        if claim.claim_id in mapping_results:
            mapping = mapping_results[claim.claim_id]
            if mapping.decision == "MATCH":
                for f in mapping.candidate_files[:3]:
                    claim.repository_evidence.append(Evidence(
                        evidence_id=f"E-MAP-{claim.claim_id}",
                        type=EvidenceType.REPOSITORY_FILE,
                        source="repository",
                        location=f,
                        content=mapping.reason,
                        supports=[claim.claim_id],
                    ))

    return {"mapping_results": mapping_results}


# ---------------------------------------------------------------------------
# 6 个独立的审计 Node — 每个 Node 是纯函数，LangGraph 调度器负责并行执行
# findings 通道为 dict + merge_findings(后写覆盖):各模块 claim_id 命名空间
# 互斥,并行同超步安全;本节点只写自己的新增条目。
# ---------------------------------------------------------------------------

def _as_dict(findings: list) -> dict[str, Any]:
    """Audit 模块返回的 finding 列表 → {claim_id: finding} dict 通道写入。"""
    return {f.claim_id: f for f in findings}


@_with_progress("audit_core_methods")
def node_audit_core_methods(state: AuditState) -> dict[str, Any]:
    """Audit Core Methods completeness (独立 Node，LangGraph 并行执行)."""
    return {"findings": _as_dict(audit_core_methods(state))}


@_with_progress("audit_training")
def node_audit_training(state: AuditState) -> dict[str, Any]:
    """Audit Training completeness (独立 Node，LangGraph 并行执行)."""
    return {"findings": _as_dict(audit_training(state))}


@_with_progress("audit_inference")
def node_audit_inference(state: AuditState) -> dict[str, Any]:
    """Audit Inference/Eval completeness (独立 Node，LangGraph 并行执行)."""
    return {"findings": _as_dict(audit_inference_eval(state))}


@_with_progress("audit_api")
def node_audit_api(state: AuditState) -> dict[str, Any]:
    """Audit API/Interface completeness (独立 Node，LangGraph 并行执行)."""
    return {"findings": _as_dict(audit_api_interface(state))}


@_with_progress("audit_resources")
def node_audit_resources(state: AuditState) -> dict[str, Any]:
    """Audit Resources completeness (独立 Node，LangGraph 并行执行)."""
    return {"findings": _as_dict(audit_resources(state))}


@_with_progress("audit_license")
def node_audit_license(state: AuditState) -> dict[str, Any]:
    """Audit License completeness (独立 Node，LangGraph 并行执行)."""
    return {"findings": _as_dict(audit_license(state))}


@_with_progress("issue_investigator")
def node_issue_investigator(state: AuditState) -> dict[str, Any]:
    """Investigate GitHub Issues/PRs for unresolved findings."""
    return {"issue_results": investigate_issues(state)}


@_with_progress("evidence_aggregator")
def node_evidence_aggregator(state: AuditState) -> dict[str, Any]:
    """Aggregate evidence from all sources."""
    return {"findings": aggregate_evidence(state, state.findings, state.issue_results)}


@_with_progress("evidence_grounder")
async def node_evidence_grounder(state: AuditState) -> dict[str, Any]:
    """Ground each code-backed finding on the actual repository code.

    取代 fusion/judge:不再用文件名列表重写 explanation,而是对每条
    「可代码验证」finding 检索 top-k 代码块原文,LLM 据代码裁决并给出
    「这段代码在做什么」的自然语言说明 + 精确位置。只回传被改动的
    finding key(增量回写),失败/降级记入 errors。
    """
    findings = list(state.findings.values())
    if not findings:
        return {}

    sem = asyncio.Semaphore(4)

    async def run_one(finding: AuditFinding) -> tuple[Any, Any]:
        async with sem:
            return await asyncio.to_thread(ground_finding, finding, state)

    outcomes = await asyncio.gather(
        *(run_one(f) for f in findings), return_exceptions=True
    )

    updates: dict[str, AuditFinding] = {}
    notes: list[str] = []
    for finding, outcome in zip(findings, outcomes):
        if isinstance(outcome, BaseException):
            notes.append(f"Evidence grounding failed for {finding.claim_id}: {outcome}")
            continue
        new_finding, note = outcome
        if new_finding is not None:
            updates[finding.claim_id] = new_finding
        if note:
            notes.append(note)

    result: dict[str, Any] = {}
    if updates:
        result["findings"] = updates
    if notes:
        result["errors"] = notes
    return result


@_with_progress("report_generator")
def node_report_generator(state: AuditState) -> dict[str, Any]:
    """Generate final report and save to project-specific directory."""
    report = generate_report(state)

    # Save reports to project-specific directory
    project_name = state.project.get("name", "unknown")
    output_dir = _get_output_dir(project_name)

    md_path = os.path.join(output_dir, "audit_report.md")
    json_path = os.path.join(output_dir, "audit_result.json")

    save_markdown_report(report, md_path)
    save_json_report(report, json_path)

    # Also save intermediate artifacts
    if state.repository_manifest:
        import json
        manifest_path = os.path.join(output_dir, "repository_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(state.repository_manifest.model_dump(), f, indent=2, default=str)

    return {"report": report}


@_with_progress("cleanup")
def node_cleanup(state: AuditState) -> dict[str, Any]:
    """Clean up cloned repository."""
    if config.audit.cleanup_repo:
        clone_path = state.project.get("clone_path", "")
        if clone_path and os.path.isdir(clone_path):
            shutil.rmtree(clone_path, ignore_errors=True)
    return {}


# ---------------------------------------------------------------------------
# Fail-fast guards: 前置条件缺失时中止，绝不基于缺失数据产出结论
# ---------------------------------------------------------------------------

def node_abort(state: AuditState) -> dict[str, Any]:
    """中止节点：记录错误并结束流程，不生成报告。

    旧实现的问题：clone 失败后流水线仍继续跑完，6 个审计节点基于空 manifest
    产出 0 findings，最终生成了看似“正常”但毫无意义的空报告。
    现在缺少前置输入时直接短路到 END，报告保持 None，由 CLI 报错退出。
    """
    return {"errors": ["Audit aborted: required inputs unavailable. No report generated."]}


def _route_has_repository(state: AuditState) -> str:
    """repository_loader 之后：仓库是否真正克隆成功？"""
    clone_path = state.project.get("clone_path", "")
    if clone_path and os.path.isdir(clone_path):
        return "continue"
    return "abort"


def _route_has_claims(state: AuditState) -> str:
    """paper_analyzer 之后：是否提取到任何 claim（审计依据）？"""
    return "continue" if state.claims else "abort"


def _route_has_manifest(state: AuditState) -> str:
    """repository_analyzer 之后：代码 manifest 是否成功生成？"""
    return "continue" if state.repository_manifest is not None else "abort"


# ---------------------------------------------------------------------------
# Build Graph
# ---------------------------------------------------------------------------

def build_workflow() -> StateGraph:
    """Build the LangGraph workflow with parallel audit fan-out.

    与旧版架构（线性链 + ThreadPoolExecutor 内部并行）相比：

    1. LangGraph 原生并行：
       - 从 planner 节点通过 6 条边并行引出 6 个审计 Node。
       - LangGraph 调度器完全感知并行状态，每个 Node 独立运行。
       - 所有审计 Node 完成后，自动汇聚到 issue_investigator。

    2. Fail-fast 守卫：
       - repository_loader / paper_analyzer / repository_analyzer 之后均设条件边，
         前置数据（仓库 / claims / manifest）缺失时短路到 abort 节点，绝不
         基于缺失数据产出空结论。

    3. 纯函数节点：
       - 每个审计 Node 只接收 AuditState，返回 findings 列表。
       - 通过 operator.add reducer 自动合并，无手动线程管理。

    4. 零 UI 依赖：
       - Graph 不包含任何 rich/console 调用。
       - 进度通过外部传入的 callback 驱动。
    """
    workflow = StateGraph(AuditState)

    # 注册所有 Node
    workflow.add_node("repository_loader", node_repository_loader)
    workflow.add_node("paper_resolver", node_paper_resolver)
    workflow.add_node("paper_analyzer", node_paper_analyzer)
    workflow.add_node("repository_analyzer", node_repository_analyzer)
    workflow.add_node("planner", node_planner)
    workflow.add_node("abort", node_abort)

    # 6 个独立审计 Node
    workflow.add_node("audit_core_methods", node_audit_core_methods)
    workflow.add_node("audit_training", node_audit_training)
    workflow.add_node("audit_inference", node_audit_inference)
    workflow.add_node("audit_api", node_audit_api)
    workflow.add_node("audit_resources", node_audit_resources)
    workflow.add_node("audit_license", node_audit_license)

    workflow.add_node("issue_investigator", node_issue_investigator)
    workflow.add_node("evidence_aggregator", node_evidence_aggregator)
    workflow.add_node("evidence_grounder", node_evidence_grounder)
    workflow.add_node("report_generator", node_report_generator)
    workflow.add_node("cleanup", node_cleanup)

    # 线性管道：前置处理（带 fail-fast 守卫）
    workflow.set_entry_point("repository_loader")
    workflow.add_conditional_edges("repository_loader", _route_has_repository, {
        "continue": "paper_resolver",
        "abort": "abort",
    })
    workflow.add_edge("paper_resolver", "paper_analyzer")
    workflow.add_conditional_edges("paper_analyzer", _route_has_claims, {
        "continue": "repository_analyzer",
        "abort": "abort",
    })
    workflow.add_conditional_edges("repository_analyzer", _route_has_manifest, {
        "continue": "planner",
        "abort": "abort",
    })
    workflow.add_edge("abort", END)

    # [关键] Fan-out: 从 Planner 并行引出 6 条边到审计 Node
    # LangGraph 自动并行执行所有目标 Node
    workflow.add_edge("planner", "audit_core_methods")
    workflow.add_edge("planner", "audit_training")
    workflow.add_edge("planner", "audit_inference")
    workflow.add_edge("planner", "audit_api")
    workflow.add_edge("planner", "audit_resources")
    workflow.add_edge("planner", "audit_license")

    # [关键] Fan-in: 所有审计 Node 完成后汇聚到 issue_investigator
    # LangGraph 等待所有前驱 Node 完成后再执行后续节点
    workflow.add_edge("audit_core_methods", "issue_investigator")
    workflow.add_edge("audit_training", "issue_investigator")
    workflow.add_edge("audit_inference", "issue_investigator")
    workflow.add_edge("audit_api", "issue_investigator")
    workflow.add_edge("audit_resources", "issue_investigator")
    workflow.add_edge("audit_license", "issue_investigator")

    # 线性管道：后续处理
    workflow.add_edge("issue_investigator", "evidence_aggregator")
    workflow.add_edge("evidence_aggregator", "evidence_grounder")
    workflow.add_edge("evidence_grounder", "report_generator")
    workflow.add_edge("report_generator", "cleanup")
    workflow.add_edge("cleanup", END)

    return workflow.compile()


async def run_audit(
    repo_url: str,
    paper_url: Optional[str] = None,
    progress_handler: Optional[BaseProgressHandler] = None,
) -> AuditState:
    """Run the full audit pipeline (async).

    Args:
        repo_url: GitHub repository URL
        paper_url: Optional paper URL
        progress_handler: Optional progress handler for UI feedback.
                          If None, uses NullProgressHandler (no output).

    Returns:
        Final AuditState with report
    """
    if progress_handler is None:
        progress_handler = NullProgressHandler()

    callback_handler = AuditCallbackHandler(progress_handler)

    initial_state = AuditState(
        project={
            "name": _get_project_name(repo_url),
            "repository_url": repo_url,
            "clone_path": "",
            "paper_url": paper_url or "",
        }
    )

    graph = build_workflow()
    # 图内包含 async 节点，必须使用 ainvoke
    final_state = await graph.ainvoke(
        initial_state,
        config={"callbacks": [callback_handler]},
    )

    llm_client.close()

    # LangGraph returns a plain dict; convert back to AuditState for attribute access
    return AuditState.model_validate(final_state)