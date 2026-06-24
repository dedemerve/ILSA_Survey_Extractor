from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF


SECTION_PATTERNS = {
    "abstract": re.compile(
        r"^\s*(?:abstract|summary)\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    "introduction": re.compile(
        r"^\s*(?:1\.?\s+)?introduction\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    "methods": re.compile(
        r"^\s*(?:\d\.?\s+)?(?:methods?|methodology|materials\s+and\s+methods|"
        r"research\s+design|research\s+methodology|data\s+and\s+methods)\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    "results": re.compile(
        r"^\s*(?:\d\.?\s+)?(?:results?|findings?|analysis)\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    "discussion": re.compile(
        r"^\s*(?:\d\.?\s+)?(?:discussion|discussion\s+and\s+conclusions?)\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    "conclusion": re.compile(
        r"^\s*(?:\d\.?\s+)?(?:conclusions?|concluding\s+remarks)\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    "references": re.compile(
        r"^\s*(?:references?|bibliography|works\s+cited)\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
}


@dataclass
class ProcessedPDF:
    """Container for the parsed content of a single PDF document."""

    file_path: Path
    file_name: str
    source_database: str
    total_pages: int
    raw_text: str
    sections: dict[str, str] = field(default_factory=dict)
    used_smart_sections: bool = False
    extraction_text: str = ""
    estimated_tokens: int = 0
    parse_errors: list[str] = field(default_factory=list)
    metadata: dict[str, Optional[str]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize processed PDF metadata for logging or persistence."""
        return {
            "file_name": self.file_name,
            "source_database": self.source_database,
            "total_pages": self.total_pages,
            "sections_found": list(self.sections.keys()),
            "used_smart_sections": self.used_smart_sections,
            "estimated_tokens": self.estimated_tokens,
            "parse_errors": self.parse_errors,
        }


def extract_raw_text(pdf_path: Path) -> tuple[str, int, list[str]]:
    """
    Extract raw text from a PDF file using PyMuPDF.

    Returns a tuple of (full_text, page_count, errors). Errors are collected
    rather than raised so that batch processing can continue.
    """
    errors: list[str] = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        errors.append(f"Failed to open PDF: {exc}")
        return "", 0, errors

    text_parts: list[str] = []
    page_count = doc.page_count
    for page_index in range(page_count):
        try:
            page = doc.load_page(page_index)
            text_parts.append(page.get_text("text"))
        except Exception as exc:
            errors.append(f"Failed to extract page {page_index}: {exc}")
    doc.close()
    return "\n".join(text_parts), page_count, errors


def detect_sections(raw_text: str) -> dict[str, tuple[int, int]]:
    """
    Detect IMRaD section boundaries within the raw text.

    Returns a dict mapping section name to (start_offset, end_offset) tuples.
    The end_offset of a section is the start_offset of the next section.
    Returns an empty dict if fewer than 3 distinct sections are reliably detected.
    """
    matches: list[tuple[str, int]] = []
    for section_name, pattern in SECTION_PATTERNS.items():
        for match in pattern.finditer(raw_text):
            matches.append((section_name, match.start()))

    if len(matches) < 3:
        return {}

    matches.sort(key=lambda pair: pair[1])

    seen: set[str] = set()
    deduped: list[tuple[str, int]] = []
    for name, start in matches:
        if name not in seen:
            deduped.append((name, start))
            seen.add(name)

    if len(deduped) < 3:
        return {}

    boundaries: dict[str, tuple[int, int]] = {}
    for index, (name, start) in enumerate(deduped):
        end = deduped[index + 1][1] if index + 1 < len(deduped) else len(raw_text)
        boundaries[name] = (start, end)
    return boundaries


MAX_CHARS = 400_000  # ~100k tokens; keeps total under gpt-4o's 128k context limit


def build_extraction_text(
    raw_text: str,
    sections: dict[str, tuple[int, int]],
) -> tuple[str, bool]:
    """
    Build the text payload that will be sent to the LLM.

    If three or more IMRaD sections are detected, concatenate the relevant
    sections (abstract through conclusion) and exclude references. Otherwise,
    fall back to the full raw text with the references tail removed if
    detectable.

    Returns (extraction_text, used_smart_sections).
    Text is hard-capped at MAX_CHARS to stay within the model context limit.
    """
    informative_keys = ("abstract", "introduction", "methods", "results", "discussion", "conclusion")

    if sections:
        parts: list[str] = []
        for key in informative_keys:
            if key in sections:
                start, end = sections[key]
                parts.append(raw_text[start:end].strip())
        if parts:
            return "\n\n".join(parts)[:MAX_CHARS], True

    references_match = SECTION_PATTERNS["references"].search(raw_text)
    if references_match:
        return raw_text[: references_match.start()].strip()[:MAX_CHARS], False
    return raw_text.strip()[:MAX_CHARS], False


def estimate_token_count(text: str) -> int:
    """
    Rough estimate of token count using a 4-characters-per-token heuristic.

    Accurate enough for batch budgeting; replace with tiktoken for precise
    accounting once a model is fixed.
    """
    return max(1, len(text) // 4)


_DOI_URL_PATTERN = re.compile(
    r"(?:https?://)?(?:dx\.)?doi\.org/(10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+)",
    re.IGNORECASE,
)
_DOI_LABEL_PATTERN = re.compile(
    r"(?:\bDOI\s*[:#]?\s*|\bdoi\s*[:#]?\s*)(10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+)",
    re.IGNORECASE,
)
_DOI_BARE_PATTERN = re.compile(
    r"\b(10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+)",
    re.IGNORECASE,
)


def _clean_doi_token(doi: str) -> str:
    """Strip trailing punctuation and URL debris from a DOI candidate."""
    doi = doi.strip()
    for prefix in (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
    ):
        if doi.lower().startswith(prefix):
            doi = doi[len(prefix):]
    return doi.rstrip(".,;:)>]}")


def extract_dois_from_text(text: str) -> list[str]:
    """
    Find DOI candidates in text. Order: doi.org URLs, labeled DOI lines, bare 10.x/…
    """
    if not text:
        return []

    found: list[str] = []
    seen: set[str] = set()

    def _add(match: re.Match[str], group: int = 1) -> None:
        raw = match.group(group)
        doi = _clean_doi_token(raw)
        key = doi.lower()
        if doi.startswith("10.") and key not in seen and len(doi) >= 12:
            seen.add(key)
            found.append(doi)

    for pattern in (_DOI_URL_PATTERN, _DOI_LABEL_PATTERN, _DOI_BARE_PATTERN):
        for match in pattern.finditer(text):
            _add(match)

    return found


def extract_doi_from_document(
    raw_text: str,
    *,
    pdf_path: Optional[Path] = None,
) -> tuple[Optional[str], list[str]]:
    """
    Aggressively scan first pages, last page, and full text for DOIs.

    Returns (primary_doi, all_unique_candidates).
    """
    regions: list[str] = []

    if pdf_path is not None:
        try:
            doc = fitz.open(pdf_path)
            page_count = doc.page_count
            for idx in range(min(3, page_count)):
                regions.append(doc.load_page(idx).get_text("text"))
            if page_count > 3:
                regions.append(doc.load_page(page_count - 1).get_text("text"))
            doc.close()
        except Exception:
            pass

    if raw_text:
        regions.append(raw_text[:25_000])
        regions.append(raw_text[-10_000:])
        if len(raw_text) <= 120_000:
            regions.append(raw_text)

    all_candidates: list[str] = []
    seen: set[str] = set()
    for region in regions:
        for doi in extract_dois_from_text(region):
            key = doi.lower()
            if key not in seen:
                seen.add(key)
                all_candidates.append(doi)

    if not all_candidates:
        return None, []
    return all_candidates[0], all_candidates


def extract_title(doc: fitz.Document) -> Optional[str]:
    """
    Multi-layer title extraction with fallback chain.
    Priority: PDF metadata (decoded) > First page heuristic > None
    """
    # Layer 1: PDF metadata (fastest, but often missing or HTML-encoded)
    metadata_title = doc.metadata.get('title', '').strip()
    if metadata_title:
        # Decode HTML entities (e.g., &amp; → &)
        metadata_title = html.unescape(metadata_title)
        if len(metadata_title) > 10:
            return metadata_title

    # Layer 2: First page heuristic
    first_page = doc[0].get_text()

    # Pattern 1: Look for title before "Abstract"
    if "Abstract" in first_page or "ABSTRACT" in first_page:
        before_abstract = first_page.split("Abstract")[0].split("ABSTRACT")[0]
        lines = [l.strip() for l in before_abstract.split('\n') if l.strip()]
        candidates = [
            l for l in lines
            if 20 <= len(l) <= 200
            and l[0].isupper()
            and not l.startswith('http')
            and not l.startswith('DOI')
            and not l.startswith('Available')
        ]
        if candidates:
            title = max(candidates, key=len)
            return title

    # Pattern 2: Look for title before author names
    author_markers = ['@', 'University', 'Department', 'School of', 'Institute']
    lines = [l.strip() for l in first_page.split('\n')[:40] if l.strip()]

    for i, line in enumerate(lines):
        if any(marker in line for marker in author_markers):
            candidates = lines[max(0, i-5):i]
            valid = [c for c in candidates if 20 <= len(c) <= 200 and c[0].isupper()]
            if valid:
                return max(valid, key=len)

    # Layer 3: No title found
    return None


def process_pdf(pdf_path: Path, source_database: str) -> ProcessedPDF:
    """
    Run the full PDF processing pipeline on a single file.

    Steps: open and extract raw text, detect IMRaD sections, build the
    extraction text payload, and estimate token usage. All errors are
    collected on the returned object; this function does not raise on
    parse failures.
    """
    raw_text, page_count, errors = extract_raw_text(pdf_path)
    sections_with_bounds = detect_sections(raw_text)
    extraction_text, used_smart = build_extraction_text(raw_text, sections_with_bounds)

    sections_text: dict[str, str] = {}
    for name, (start, end) in sections_with_bounds.items():
        sections_text[name] = raw_text[start:end].strip()

    # Extract title and DOI with multi-layer fallback
    extracted_title = None
    extracted_doi = None
    doi_candidates: list[str] = []
    try:
        doc = fitz.open(pdf_path)
        extracted_title = extract_title(doc)
        doc.close()
    except Exception as e:
        errors.append(f"Failed to extract title: {e}")

    try:
        extracted_doi, doi_candidates = extract_doi_from_document(
            raw_text, pdf_path=pdf_path
        )
    except Exception as e:
        errors.append(f"Failed to extract DOI: {e}")

    return ProcessedPDF(
        file_path=pdf_path,
        file_name=pdf_path.name,
        source_database=source_database,
        total_pages=page_count,
        raw_text=raw_text,
        sections=sections_text,
        used_smart_sections=used_smart,
        extraction_text=extraction_text,
        estimated_tokens=estimate_token_count(extraction_text),
        parse_errors=errors,
        metadata={
            "extracted_title": extracted_title,
            "extracted_doi": extracted_doi,
            "doi_candidates": doi_candidates,
        },
    )


def discover_pdfs(raw_pdfs_root: Path) -> list[tuple[Path, str]]:
    """
    Walk the raw_pdfs root directory and return all PDFs paired with their
    source database name (the immediate parent folder name).

    Expected layout:
        raw_pdfs_root/
            wos/*.pdf
            scopus/*.pdf
            oecd/*.pdf
            iea/*.pdf
    """
    valid_sources = {"wos", "scopus", "oecd", "iea"}
    discovered: list[tuple[Path, str]] = []
    for source_dir in raw_pdfs_root.iterdir():
        if not source_dir.is_dir():
            continue
        source_name = source_dir.name.lower()
        if source_name not in valid_sources:
            continue
        for pdf_file in source_dir.rglob("*.pdf"):
            discovered.append((pdf_file, source_name))
    return discovered


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    raw_pdfs_root = project_root / "data" / "raw_pdfs"

    pdfs = discover_pdfs(raw_pdfs_root)
    print(f"Discovered {len(pdfs)} PDFs across all source databases.")

    if not pdfs:
        print("No PDFs found. Place files under data/raw_pdfs/{wos,scopus,oecd,iea}/ to test.")
    else:
        sample_path, sample_source = pdfs[0]
        print(f"\nProcessing sample: {sample_path.name} (source: {sample_source})")
        result = process_pdf(sample_path, sample_source)
        print(f"\nProcessing summary: {result.to_dict()}")
        preview_chars = 500
        print(f"\nExtraction text preview (first {preview_chars} chars):")
        print(result.extraction_text[:preview_chars])
        print(f"\n...")
        print(f"\nTotal extraction text length: {len(result.extraction_text)} chars")
        print(f"Estimated tokens: {result.estimated_tokens}")
        if result.parse_errors:
            print(f"\nParse errors encountered: {result.parse_errors}")