"""
Data models for research papers and pipeline state.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class Author(BaseModel):
    """Author of a research paper."""
    name: str
    author_id: Optional[str] = None


class Paper(BaseModel):
    """Structured representation of a research paper."""
    paper_id: str
    title: str
    authors: List[Author] = Field(default_factory=list)
    year: Optional[int] = None
    abstract: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    url: Optional[str] = None
    citation_count: Optional[int] = None
    references: List[str] = Field(default_factory=list)
    source: str = "semantic_scholar"
    confidence: float = 1.0
    extraction_notes: Optional[str] = None


class PipelineState(BaseModel):
    """
    Shared state across all LangGraph nodes.
    Tracks papers, processing status, and traversal metadata.
    """
    seed_paper_url: str = ""
    current_hop: int = 0
    max_hops: int = 2

    # Paper collections
    papers_to_fetch: List[str] = Field(default_factory=list)
    fetched_raw: List[Dict[str, Any]] = Field(default_factory=list)
    parsed_papers: List[Paper] = Field(default_factory=list)
    extracted_papers: List[Paper] = Field(default_factory=list)
    reviewed_papers: List[Paper] = Field(default_factory=list)
    deduplicated_papers: List[Paper] = Field(default_factory=list)
    stored_papers: List[Paper] = Field(default_factory=list)

    # Tracking
    all_processed_ids: List[str] = Field(default_factory=list)
    duplicate_count: int = 0
    low_confidence_count: int = 0
    total_fetched: int = 0
    errors: List[str] = Field(default_factory=list)

    # Human-in-the-loop
    needs_human_review: List[Paper] = Field(default_factory=list)
    human_reviewed: bool = False
