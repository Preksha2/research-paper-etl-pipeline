"""
FastAPI server exposing the citation analysis pipeline
as a REST endpoint.
"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.schemas import PipelineRequest, PipelineResponse, PaperSummary, HealthResponse
from src.graph.pipeline import run_pipeline
from src.utils.config import load_env


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup."""
    try:
        load_env()
        logger.info("API server started, environment loaded")
    except Exception as e:
        logger.error(f"Startup error: {e}")
    yield
    logger.info("API server shutting down")


app = FastAPI(
    title="Research Paper ETL Pipeline",
    description="AI-powered citation chain traversal and analysis using LangGraph",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and service connectivity."""
    openai_ok = bool(os.getenv("OPENAI_API_KEY"))
    pinecone_ok = bool(os.getenv("PINECONE_API_KEY"))

    return HealthResponse(
        status="healthy",
        pinecone_connected=pinecone_ok,
        openai_configured=openai_ok,
    )


@app.post("/run", response_model=PipelineResponse)
async def run_etl_pipeline(request: PipelineRequest):
    """
    Start a full citation traversal pipeline.

    Accepts a seed paper URL/ID and traverses its citation chain
    up to max_hops deep, returning a structured dataset.
    """
    logger.info(f"Pipeline request: seed={request.seed_paper_url}, hops={request.max_hops}")

    try:
        final_state = run_pipeline(
            seed_paper_url=request.seed_paper_url,
            max_hops=request.max_hops,
        )

        # Build paper summaries
        stored = final_state.get("stored_papers", [])
        paper_summaries = []
        for p in stored:
            paper_summaries.append(PaperSummary(
                paper_id=p.paper_id if hasattr(p, 'paper_id') else p.get("paper_id", ""),
                title=p.title if hasattr(p, 'title') else p.get("title", ""),
                authors=", ".join(a.name for a in p.authors) if hasattr(p, 'authors') else str(p.get("authors", "")),
                year=p.year if hasattr(p, 'year') else p.get("year"),
                doi=p.doi if hasattr(p, 'doi') else p.get("doi"),
                citation_count=p.citation_count if hasattr(p, 'citation_count') else p.get("citation_count"),
                confidence=p.confidence if hasattr(p, 'confidence') else p.get("confidence", 0),
            ))

        return PipelineResponse(
            status="completed",
            seed_paper=request.seed_paper_url,
            total_fetched=final_state.get("total_fetched", 0),
            papers_stored=len(stored),
            duplicates_caught=final_state.get("duplicate_count", 0),
            low_confidence_flagged=final_state.get("low_confidence_count", 0),
            errors=len(final_state.get("errors", [])),
            papers=paper_summaries,
        )

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")
