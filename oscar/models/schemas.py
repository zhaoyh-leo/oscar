"""Core data models for OSCAR.

Defines the Claim → Evidence → Judgment pipeline data structures.
"""

from __future__ import annotations

import operator
from enum import Enum
from typing import Annotated, Any, Literal, Optional
from pydantic import BaseModel, Field


class ClaimStatus(str, Enum):
    """Status of a claim after evaluation."""
    VERIFIED = "VERIFIED"
    INCOMPLETE = "INCOMPLETE"
    MISSING = "MISSING"
    RESTRICTED = "RESTRICTED"
    PLANNED = "PLANNED"
    UNCERTAIN = "UNCERTAIN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class EvidenceType(str, Enum):
    """Type of evidence."""
    REPOSITORY_FILE = "repository_file"
    REPOSITORY_STRUCTURE = "repository_structure"
    README = "readme"
    PAPER = "paper"
    ISSUE = "issue"
    PR = "pr"
    LICENSE = "license"
    EXTERNAL_RESOURCE = "external_resource"
    LLM_INFERENCE = "llm_inference"


class ClaimCategory(str, Enum):
    """Category of a claim."""
    CORE_METHOD = "core_method"
    TRAINING = "training"
    INFERENCE = "inference"
    EVALUATION = "evaluation"
    DEMO = "demo"
    BENCHMARK = "benchmark"
    API_INTERFACE = "api_interface"
    DATASET = "dataset"
    CHECKPOINT = "checkpoint"
    EXTERNAL_RESOURCE = "external_resource"
    LICENSE = "license"
    IMPLEMENTATION = "implementation"
    RELEASE = "release"


class ClaimSource(BaseModel):
    """Source of a claim."""
    type: str = Field(description="Source type: paper, readme, or repository")
    location: str = Field(description="Location within the source, e.g. 'Section 3.2'")
    content: Optional[str] = Field(None, description="Extracted snippet")


class Evidence(BaseModel):
    """A piece of evidence supporting or refuting a claim."""
    evidence_id: str = Field(description="Unique evidence identifier, e.g. E-001")
    type: EvidenceType = Field(description="Type of evidence")
    source: str = Field(description="Source identifier, e.g. 'github', 'paper', 'arxiv'")
    location: str = Field(description="Location of evidence, e.g. file path, URL, line number")
    content: str = Field(description="Content or summary of evidence")
    supports: list[str] = Field(default_factory=list, description="Claim IDs this evidence supports")
    contradicts: list[str] = Field(default_factory=list, description="Claim IDs this evidence contradicts")


class Claim(BaseModel):
    """A claim extracted from paper or README that needs to be verified."""
    claim_id: str = Field(description="Unique claim identifier, e.g. METHOD-001")
    category: ClaimCategory = Field(description="Category of the claim")
    statement: str = Field(description="The claim statement")
    source: ClaimSource = Field(description="Source of the claim")
    expected_evidence: list[str] = Field(default_factory=list, description="Expected evidence types")
    repository_evidence: list[Evidence] = Field(default_factory=list, description="Evidence from repository")
    external_evidence: list[Evidence] = Field(default_factory=list, description="Evidence from external sources")
    status: ClaimStatus = Field(default=ClaimStatus.UNCERTAIN, description="Current evaluation status")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence in the status")
    explanation: str = Field(default="", description="Explanation of the evaluation")


class MappingResult(BaseModel):
    """Result of mapping a claim to code."""
    claim_id: str = Field(description="Claim ID being mapped")
    candidate_files: list[str] = Field(default_factory=list, description="Candidate file paths")
    candidate_classes: list[str] = Field(default_factory=list, description="Candidate class names")
    candidate_functions: list[str] = Field(default_factory=list, description="Candidate function names")
    decision: str = Field(default="UNMATCHED", description="MATCH or UNMATCHED")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence in the mapping")
    reason: str = Field(default="", description="Reason for the mapping decision")


class CodeLocation(BaseModel):
    """Precise code location with line numbers."""
    module_path: str = Field(description="Relative file path")
    line_start: int = Field(default=0, description="Start line number")
    line_end: int = Field(default=0, description="End line number")
    class_name: Optional[str] = Field(None, description="Containing class, if any")
    function_name: Optional[str] = Field(None, description="Function or method name")
    snippet: str = Field(default="", description="Relevant code snippet")


class RepositoryManifest(BaseModel):
    """Repository manifest - structural overview of the repository."""
    files: list[str] = Field(default_factory=list)
    directories: list[str] = Field(default_factory=list)
    python_modules: list[str] = Field(default_factory=list)
    classes: dict[str, list[str]] = Field(default_factory=dict, description="module_path -> [class names]")
    class_locations: dict[str, list[CodeLocation]] = Field(default_factory=dict, description="module_path -> [CodeLocation]")
    functions: dict[str, list[str]] = Field(default_factory=dict, description="module_path -> [function names]")
    function_locations: dict[str, list[CodeLocation]] = Field(default_factory=dict, description="module_path -> [CodeLocation]")
    configs: list[str] = Field(default_factory=list)
    scripts: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    licenses: list[str] = Field(default_factory=list)
    entry_points: list[str] = Field(default_factory=list)
    stub_files: list[str] = Field(default_factory=list, description="Files containing stubs")
    stub_details: dict[str, list[str]] = Field(default_factory=dict, description="file_path -> [stub descriptions]")
    code_summary: Optional[CodeSummary] = Field(None, description="Multi-level code summary")


class PaperSection(BaseModel):
    """A single section extracted from a paper."""
    section_id: str = Field(description="Section identifier, e.g. 'sec-1'")
    title: str = Field(description="Section title")
    content: str = Field(description="Section content text")
    page_start: int = Field(default=0, description="Starting page number")
    page_end: int = Field(default=0, description="Ending page number")


class PaperChunk(BaseModel):
    """A single chunk of paper text with metadata."""
    chunk_id: str = Field(description="Unique chunk identifier")
    section: str = Field(description="Source section name")
    text: str = Field(description="Chunk content")
    char_offset: int = Field(default=0, description="Character offset in original text")


class PaperInfo(BaseModel):
    """Information about a paper."""
    title: Optional[str] = None
    authors: list[str] = Field(default_factory=list)
    abstract: Optional[str] = None
    pdf_url: Optional[str] = None
    pdf_path: Optional[str] = None
    arxiv_id: Optional[str] = None
    sections: dict[str, str] = Field(default_factory=dict, description="section_name -> content")
    full_text: Optional[str] = None
    chunks: list[PaperChunk] = Field(default_factory=list, description="Overlapping chunks for Map-Reduce processing")


class IssueResult(BaseModel):
    """Result from GitHub Issues/PR investigation."""
    issue_number: int
    title: str
    body: Optional[str] = None
    state: str = "open"
    author: Optional[str] = None
    is_pr: bool = False
    relevant_claims: list[str] = Field(default_factory=list)
    key_statements: list[str] = Field(default_factory=list, description="Key statements extracted from the issue")
    explanation: Optional[str] = None
    status_implication: Optional[str] = Field(
        default=None,
        description="LLM's structured implication: PLANNED / RESTRICTED / MISSING / UNCERTAIN ("
        "never inferred by substring-scanning prose — negations like 'no copyright "
        "restriction' would be misread)",
    )


class EvidenceDetail(BaseModel):
    """Granular evidence detail with precise location (file, line, class, method)."""
    file_path: str = Field(description="File path relative to repo root")
    line_number: int = Field(default=0, description="Start line number in the file")
    line_end: int = Field(default=0, description="End line number in the file")
    class_name: Optional[str] = Field(None, description="Containing class name, if any")
    function_name: Optional[str] = Field(None, description="Containing function/method name, if any")
    snippet: str = Field(default="", description="Relevant code snippet (up to 200 chars)")
    label: str = Field(default="", description="Short label, e.g. 'optimizer definition', 'dataset class'")
    code_explanation: str = Field(default="", description="Natural-language explanation of what this code does")


class GroundedLocation(BaseModel):
    """A code location identified by evidence grounding, with an NL explanation.

    ``chunk_ref`` anchors the location to one of the code chunks shown to the
    LLM (e.g. "C1"); the grounder maps it back to the chunk's authoritative
    file/symbol/line span instead of trusting LLM-provided coordinates.
    """
    chunk_ref: str = ""
    file_path: str = ""
    class_name: Optional[str] = None
    function_name: Optional[str] = None
    snippet: str = ""
    what_it_does: str = Field(default="", description="Natural-language explanation of what this code does")


class GroundingVerdict(BaseModel):
    """Structured output of LLM evidence grounding for one finding."""
    verdict: Literal["VERIFIED", "INCOMPLETE", "MISSING", "UNCERTAIN"]
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    explanation: str = Field(default="", description="Overall explanation referencing the given code")
    locations: list[GroundedLocation] = Field(default_factory=list)


class CodeSummary(BaseModel):
    """Multi-level code summary for context understanding."""
    project_summary: str = Field(default="", description="Project-level summary (what the project does)")
    file_summaries: dict[str, str] = Field(default_factory=dict, description="file_path -> one-line summary")
    class_method_summaries: dict[str, str] = Field(default_factory=dict, description="class_or_method -> one-line summary")


class AuditFinding(BaseModel):
    """A single audit finding."""
    claim_id: str
    category: ClaimCategory
    statement: str
    status: ClaimStatus
    confidence: float
    evidence_summary: str = Field(default="", description="Summary of evidence")
    evidence_details: list[EvidenceDetail] = Field(default_factory=list, description="Granular evidence with line numbers")
    explanation: str = Field(default="", description="Detailed explanation")


class AuditReport(BaseModel):
    """Final audit report."""
    project_name: str
    repository_url: str
    paper_title: Optional[str] = None
    executive_summary: str = ""
    findings: list[AuditFinding] = Field(default_factory=list)
    uncertain_findings: list[AuditFinding] = Field(default_factory=list)
    evidence_index: list[Evidence] = Field(default_factory=list)
    summary_stats: dict[str, int] = Field(default_factory=dict)
    overall_score: float = Field(default=0.0, description="Weighted completeness score (scored categories only)")
    category_stats: dict[str, dict[str, Any]] = Field(default_factory=dict, description="Per-category score stats")


def merge_findings(
    current: dict[str, AuditFinding],
    update: dict[str, AuditFinding],
) -> dict[str, AuditFinding]:
    """Findings channel reducer — later writes win per ``claim_id``.

    为什么是 dict 而不是 operator.add 的 list:
    6 个 audit 节点并行各 append 一次后,aggregator/grounder 需要「整列表
    回写」——list 的 operator.add 只会追加,导致每份 finding 在报告里出现 4 次。
    dict 通道下并行节点写互斥 key(audit 模块的 claim_id 命名空间互斥),
    后续线性节点只回传被改动的 key,reducer 按 key 后写覆盖,天然去重。
    超步(superstep)语义保证:并行 audit 节点同属一个超步、aggregator/grounder
    在其后的超步执行,「后写覆盖」顺序确定。
    """
    merged = dict(current)
    merged.update(update)
    return merged


class AuditState(BaseModel):
    """LangGraph state for the audit workflow.

    ``findings`` 使用 dict + merge_findings reducer:并行 audit 节点写互斥 key,
    后续节点按 key 后写覆盖——报告期每个 claim_id 只出现一份(旧实现 list +
    operator.add 使 aggregator/judge 把全量 findings 二次追加,每份重复 4 次)。
    """
    project: dict[str, Any] = Field(default_factory=lambda: {
        "name": "",
        "repository_url": "",
        "clone_path": "",
    })
    paper: Optional[PaperInfo] = None
    readme_content: Optional[str] = Field(None, description="Full README text")
    repository_manifest: Optional[RepositoryManifest] = None
    claims: list[Claim] = Field(default_factory=list)
    evidences: list[Evidence] = Field(default_factory=list)
    mapping_results: dict[str, MappingResult] = Field(default_factory=dict)
    findings: Annotated[dict[str, AuditFinding], merge_findings] = Field(default_factory=dict)
    issue_results: list[IssueResult] = Field(default_factory=list)
    report: Optional[AuditReport] = None
    errors: Annotated[list[str], operator.add] = Field(default_factory=list)