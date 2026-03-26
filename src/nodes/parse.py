"""
Parse node: transforms raw Semantic Scholar API responses
into structured Paper objects.
"""
from typing import List, Dict, Any
from loguru import logger

from src.models.paper import Paper, Author


def parse_raw_paper(raw: Dict[str, Any]) -> Paper:
    """
    Parse a raw Semantic Scholar API response into a Paper object.

    Args:
        raw: Raw dict from the Semantic Scholar API.

    Returns:
        Structured Paper object.
    """
    paper_id = raw.get("paperId", "unknown")
    title = raw.get("title", "Untitled")

    # Parse authors
    authors = []
    for author_data in raw.get("authors", []):
        authors.append(Author(
            name=author_data.get("name", "Unknown"),
            author_id=author_data.get("authorId"),
        ))

    # Parse external IDs
    external_ids = raw.get("externalIds") or {}
    doi = external_ids.get("DOI")
    arxiv_id = external_ids.get("ArXiv")

    # Parse references (just IDs for traversal)
    references = []
    for ref in raw.get("references", []) or []:
        ref_id = ref.get("paperId")
        if ref_id:
            references.append(ref_id)

    return Paper(
        paper_id=paper_id,
        title=title,
        authors=authors,
        year=raw.get("year"),
        abstract=raw.get("abstract"),
        doi=doi,
        arxiv_id=arxiv_id,
        url=raw.get("url"),
        citation_count=raw.get("citationCount"),
        references=references,
        source="semantic_scholar",
        confidence=1.0 if doi else 0.8,
    )


def parse_node(state: dict) -> dict:
    """
    LangGraph node: Parse raw API responses into Paper objects.

    Reads from state['fetched_raw'], parses each one,
    and stores results in state['parsed_papers'].
    """
    raw_papers = state.get("fetched_raw", [])

    if not raw_papers:
        logger.warning("Parse node: no raw papers to parse")
        state["parsed_papers"] = []
        return state

    parsed = []
    for raw in raw_papers:
        try:
            paper = parse_raw_paper(raw)
            parsed.append(paper)
            logger.debug(f"  Parsed: {paper.title[:60]} (refs: {len(paper.references)})")
        except Exception as e:
            logger.error(f"  Failed to parse paper: {e}")
            errors = state.get("errors", [])
            errors.append(f"Parse error: {e}")
            state["errors"] = errors

    state["parsed_papers"] = parsed
    logger.info(f"Parse node complete: {len(parsed)} papers parsed")
    return state
