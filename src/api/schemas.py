"""
Pydantic models for API request/response validation.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class PipelineRequest(BaseModel):
    """Request to start a pipeline run."""
    seed_paper_url: str = Field(..., description="Semantic Scholar URL, ArXiv URL, DOI, or paper ID")
    max_hops: Optional[int] = Field(2, ge=1, le=5, description="Maximum citation hops to traverse")


class PaperSummary(BaseModel):
    """Summary of a stored paper."""
    paper_id: str
    title: str
    authors: str
    year: Optional[int]
    doi: Optional[str]
    citation_count: Optional[int]
    confidence: float


class PipelineResponse(BaseModel):
    """Response after pipeline completion."""
    status: str
    seed_paper: str
    total_fetched: int
    papers_stored: int
    duplicates_caught: int
    low_confidence_flagged: int
    errors: int
    papers: List[PaperSummary]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    pinecone_connected: bool
    openai_configured: bool
