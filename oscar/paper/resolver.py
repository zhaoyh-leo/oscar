"""Paper resolver - find, download, and extract text from paper."""

import os
import re
from pathlib import Path
from typing import Optional

import httpx

from oscar.config import config
from oscar.models.schemas import PaperInfo, PaperSection
from oscar.utils.github import fetch_repo_metadata, parse_github_url
from oscar.utils.cache import paper_cache_get, paper_cache_set, paper_cache_has_text, paper_cache_set_text


def resolve_paper(
    repo_url: str,
    paper_url: Optional[str] = None,
    paper_pdf: Optional[str] = None,
) -> Optional[PaperInfo]:
    """Resolve paper information, download PDF, and extract full text.

    Priority:
    1. Direct paper PDF URL / file path
    2. arXiv URL (download PDF from arXiv, extract text)
    3. Repository metadata (description, README links)
    """
    if paper_url:
        info = _resolve_from_url(paper_url)
    elif paper_pdf:
        info = _resolve_from_pdf_path(paper_pdf)
    else:
        info = _resolve_from_repo(repo_url)

    # Download PDF and extract text if we have a valid URL
    if info and info.arxiv_id:
        # Check cache for already-extracted text
        cached_text_path = paper_cache_has_text(info.arxiv_id)
        if cached_text_path:
            with open(cached_text_path, "r", encoding="utf-8") as f:
                info.full_text = f.read()
            _parse_sections(info)
            return info

    if not info:
        return None

    pdf_path = _download_pdf(info)
    if pdf_path:
        _extract_text_with_pymupdf(info, pdf_path)
        _parse_sections(info)
        # Cache extracted text
        if info.arxiv_id and info.full_text:
            paper_cache_set_text(info.arxiv_id, info.full_text)

    return info


def _resolve_from_url(url: str) -> Optional[PaperInfo]:
    """Resolve paper from URL, extracting arXiv metadata."""
    info = PaperInfo()
    info.pdf_url = url

    arxiv_match = re.search(r"arxiv\.org/(?:abs|pdf)/(\d+\.\d+)", url)
    if arxiv_match:
        arxiv_id = arxiv_match.group(1)
        info.arxiv_id = arxiv_id
        info.pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        try:
            api_url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
            resp = httpx.get(api_url, timeout=15)
            if resp.status_code == 200:
                # Parse Atom XML - get the title from inside <entry>, not <feed>
                text = resp.text
                # Find the entry title (skip the feed title)
                entry_match = re.search(r"<entry>.*?<title>(.*?)</title>", text, re.DOTALL)
                if entry_match:
                    info.title = entry_match.group(1).strip().replace("\n", " ").replace("  ", " ")
                abstract_match = re.search(r"<summary>(.*?)</summary>", text, re.DOTALL)
                if abstract_match:
                    info.abstract = abstract_match.group(1).strip().replace("\n", " ").replace("  ", " ")
        except Exception:
            pass

    return info


def _resolve_from_pdf_path(path: str) -> Optional[PaperInfo]:
    """Resolve paper from a local PDF file path."""
    info = PaperInfo()
    info.pdf_url = path
    return info


def _resolve_from_repo(repo_url: str) -> Optional[PaperInfo]:
    """Try to resolve paper from repository metadata."""
    metadata = fetch_repo_metadata(repo_url)
    if not metadata:
        return None

    info = PaperInfo()
    if metadata.get("description"):
        desc = metadata["description"]
        arxiv_match = re.search(r"arxiv\.org/(?:abs|pdf)/(\d+\.\d+)", desc)
        if arxiv_match:
            return _resolve_from_url(f"https://arxiv.org/abs/{arxiv_match.group(1)}")

    return info if info.title or info.abstract else None


def _download_pdf(info: PaperInfo) -> Optional[str]:
    """Download paper PDF and save to project-specific paper directory.

    Returns the local file path, or None if download fails.
    """
    if not info.pdf_url:
        return None

    # Determine project name from title or arxiv_id
    project_name = info.arxiv_id or "paper"
    if info.title:
        import re
        project_name = re.sub(r"[^a-zA-Z0-9_-]", "_", info.title.split(":")[0].split(",")[0].strip()[:50]).strip("_")
        if not project_name:
            project_name = info.arxiv_id or "paper"
    pdf_url = info.pdf_url

    # Only download from known sources
    if not ("arxiv.org" in pdf_url or pdf_url.endswith(".pdf")):
        return None

    try:
        paper_dir = config.paths.output_dir / "papers" / project_name
        paper_dir.mkdir(parents=True, exist_ok=True)

        # Determine filename
        if info.arxiv_id:
            pdf_path = paper_dir / f"{info.arxiv_id}.pdf"
        else:
            pdf_path = paper_dir / "paper.pdf"

        if pdf_path.exists():
            return str(pdf_path)

        # Download PDF
        headers = {"User-Agent": "OSCAR/0.1 (research audit tool)"}
        resp = httpx.get(pdf_url, headers=headers, follow_redirects=True, timeout=60)
        if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("application/pdf"):
            pdf_path.write_bytes(resp.content)
            return str(pdf_path)

    except Exception:
        pass
    return None


def _extract_text_with_pymupdf(info: PaperInfo, pdf_path: str):
    """Extract full text from PDF using PyMuPDF."""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        pages_text = []
        for page in doc:
            pages_text.append(page.get_text())
        doc.close()

        info.full_text = "\n\n".join(pages_text)
    except ImportError:
        pass
    except Exception:
        pass


def _parse_sections(info: PaperInfo):
    """Parse full text into sections (Introduction, Method, Experiments, etc.)."""
    if not info.full_text:
        return

    text = info.full_text
    lines = text.split("\n")

    # Common section headers in academic papers
    section_patterns = [
        r"^\s*(?:I\.?\s*)?(?:Introduction|BACKGROUND)\s*$",
        r"^\s*(?:II\.?\s*)?(?:Related Work|RELATED WORK)\s*$",
        r"^\s*(?:III\.?\s*)?(?:Method|Approach|Proposed|Model|Framework|System)\s*$",
        r"^\s*(?:IV\.?\s*)?(?:Experiment|Evaluation|Results|Benchmark)\s*$",
        r"^\s*(?:V\.?\s*)?(?:Discussion|Analysis|Ablation)\s*$",
        r"^\s*(?:VI\.?\s*)?(?:Conclusion|Limitation|Future Work)\s*$",
        r"^\s*(?:Appendix|Supplementary|REFERENCES)\s*$",
    ]

    # Also grab numbered sections: 1.1, 3.2, etc.
    numbered_section = re.compile(r"^\s*(\d+(?:\.\d+)?)\s+([A-Z][A-Za-z0-9\s/-]+)$")

    sections = {}
    current_section = "Abstract"
    current_content = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Check for section headers
        matched = False
        for pat in section_patterns:
            if re.match(pat, stripped, re.IGNORECASE):
                if current_content:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = stripped
                current_content = []
                matched = True
                break

        if matched:
            continue

        # Check for numbered sections
        m = numbered_section.match(stripped)
        if m and len(stripped) < 100:
            if current_content:
                sections[current_section] = "\n".join(current_content).strip()
            current_section = stripped
            current_content = []
            continue

        current_content.append(stripped)

    # Save last section
    if current_content:
        sections[current_section] = "\n".join(current_content).strip()

    info.sections = sections


def chunk_paper(
    paper: Optional[PaperInfo],
    chunk_size: int = 2500,
    overlap: int = 250,
) -> Optional[PaperInfo]:
    """Split paper sections into overlapping chunks.

    Each chunk is a self-contained segment of paper text (2500 chars by default)
    with 10% overlap between adjacent chunks to prevent semantic breaks at
    chunk boundaries (e.g. method names split across chunks).

    Args:
        paper: The paper to chunk.
        chunk_size: Target size per chunk in characters.
        overlap: Overlap between consecutive chunks in characters.

    Returns:
        The same PaperInfo with paper.chunks populated, or None if no text.
    """
    if not paper or not paper.sections:
        return paper

    from oscar.models.schemas import PaperChunk

    chunks: list[PaperChunk] = []
    chunk_counter = [0]

    for section_name, section_content in paper.sections.items():
        if not section_content or len(section_content.strip()) < 50:
            continue

        paragraphs = section_content.split("\n\n")
        current_text = ""
        current_offset = _find_section_offset(paper.full_text or "", section_content)

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_text) + len(para) + 2 <= chunk_size:
                current_text += para + "\n\n"
            else:
                if current_text.strip():
                    chunk_counter[0] += 1
                    chunks.append(PaperChunk(
                        chunk_id=f"chunk-{chunk_counter[0]:03d}",
                        section=section_name,
                        text=current_text.strip(),
                        char_offset=current_offset,
                    ))
                current_text = para + "\n\n"
                current_offset += len(current_text)  # approximate

        if current_text.strip():
            chunk_counter[0] += 1
            chunks.append(PaperChunk(
                chunk_id=f"chunk-{chunk_counter[0]:03d}",
                section=section_name,
                text=current_text.strip(),
                char_offset=current_offset,
            ))

    # Apply overlap: append the first `overlap` chars of the next chunk
    # to the current chunk so that sentences/method names at boundaries
    # are not lost.
    if overlap > 0 and len(chunks) > 1:
        for i in range(len(chunks) - 1):
            next_prefix = chunks[i + 1].text[:overlap]
            chunks[i] = PaperChunk(
                chunk_id=chunks[i].chunk_id,
                section=chunks[i].section,
                text=chunks[i].text + "\n\n" + next_prefix,
                char_offset=chunks[i].char_offset,
            )

    paper.chunks = chunks
    return paper


def _find_section_offset(full_text: str, section_content: str) -> int:
    """Find the approximate character offset of a section in the full text."""
    if not full_text or not section_content:
        return 0
    # Use the first 50 chars of the section as a search anchor
    anchor = section_content[:50].strip()
    idx = full_text.find(anchor)
    return idx if idx >= 0 else 0