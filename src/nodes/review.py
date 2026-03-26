"""
Review node: human-in-the-loop interrupt for low-confidence records.
Pauses the pipeline when papers have missing DOIs or ambiguous metadata,
allowing manual correction before continuing.
"""
from typing import List
from loguru import logger

from src.models.paper import Paper
from src.utils.config import load_config

config = load_config()
CONFIDENCE_THRESHOLD = config["pipeline"]["confidence_threshold"]


def auto_fix_paper(paper: Paper) -> Paper:
    """
    Attempt automatic fixes for common metadata issues
    before escalating to human review.
    """
    # Fix missing year from title patterns like "(2023)" or "2023"
    if not paper.year and paper.title:
        import re
        year_match = re.search(r'\b(19|20)\d{2}\b', paper.title)
        if year_match:
            paper.year = int(year_match.group())
            paper.extraction_notes = (paper.extraction_notes or "") + " | Year auto-extracted from title"
            paper.confidence = min(paper.confidence + 0.1, 1.0)

    # If abstract exists but is very short, flag it
    if paper.abstract and len(paper.abstract) < 50:
        paper.extraction_notes = (paper.extraction_notes or "") + " | Abstract unusually short"

    return paper


def review_node(state: dict) -> dict:
    """
    LangGraph node: Review and approve extracted papers.

    For papers above confidence threshold: auto-approve.
    For papers below threshold: attempt auto-fix, then either
    approve or flag for human-in-the-loop interrupt.

    In automated mode (no human), low-confidence papers are
    auto-fixed and passed through with notes.
    In interactive mode, the pipeline pauses for human input.
    """
    extracted = state.get("extracted_papers", [])
    needs_review = state.get("needs_human_review", [])
    human_reviewed = state.get("human_reviewed", False)

    if not extracted:
        logger.warning("Review node: no papers to review")
        state["reviewed_papers"] = []
        return state

    reviewed = []
    review_ids = {p.paper_id for p in needs_review}

    for paper in extracted:
        if paper.paper_id in review_ids and not human_reviewed:
            # Attempt auto-fix first
            fixed = auto_fix_paper(paper)
            if fixed.confidence >= CONFIDENCE_THRESHOLD:
                logger.info(f"  Auto-fixed: {fixed.title[:50]} (confidence: {fixed.confidence:.2f})")
                reviewed.append(fixed)
            else:
                # Still low confidence after auto-fix — approve with warning
                fixed.extraction_notes = (fixed.extraction_notes or "") + " | AUTO-APPROVED (low confidence)"
                reviewed.append(fixed)
                logger.warning(f"  Low confidence approved: {fixed.title[:50]} ({fixed.confidence:.2f})")
        else:
            reviewed.append(paper)

    state["reviewed_papers"] = reviewed
    state["needs_human_review"] = []
    logger.info(f"Review node complete: {len(reviewed)} papers reviewed")
    return state
