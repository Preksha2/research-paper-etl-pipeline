"""
Store node: persists deduplicated papers into Pinecone with vector embeddings.
Each paper is stored with its embedding and full metadata for later retrieval.
"""
import os
import time
from typing import List, Dict, Any
from loguru import logger
from pinecone import Pinecone
from openai import OpenAI

from src.models.paper import Paper
from src.nodes.deduplicate import get_pinecone_client, ensure_index_exists, get_embedding
from src.utils.config import load_config

config = load_config()
PINECONE_CONFIG = config["pinecone"]
BATCH_SIZE = config["pipeline"]["batch_size"]


def paper_to_text(paper: Paper) -> str:
    """Create a text representation for embedding."""
    parts = [paper.title]
    if paper.abstract:
        parts.append(paper.abstract[:500])
    if paper.authors:
        parts.append("Authors: " + ", ".join(a.name for a in paper.authors[:5]))
    if paper.year:
        parts.append(f"Year: {paper.year}")
    return " | ".join(parts)


def paper_to_metadata(paper: Paper) -> Dict[str, Any]:
    """Convert paper to Pinecone metadata dict (must be flat)."""
    return {
        "title": paper.title[:500],
        "year": paper.year or 0,
        "doi": paper.doi or "",
        "arxiv_id": paper.arxiv_id or "",
        "citation_count": paper.citation_count or 0,
        "authors": ", ".join(a.name for a in paper.authors[:5]),
        "confidence": paper.confidence,
        "source": paper.source,
        "num_references": len(paper.references),
        "url": paper.url or "",
    }


def store_node(state: dict) -> dict:
    """
    LangGraph node: Store papers in Pinecone with embeddings.

    Generates embeddings via OpenAI, then upserts papers into Pinecone
    in batches. Each paper is stored with full metadata for retrieval.
    """
    papers = state.get("deduplicated_papers", [])

    if not papers:
        logger.warning("Store node: no papers to store")
        state["stored_papers"] = []
        return state

    pc = get_pinecone_client()
    index_name = os.getenv("PINECONE_INDEX_NAME", PINECONE_CONFIG["index_name"])
    ensure_index_exists(pc, index_name)
    index = pc.Index(index_name)

    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    stored = []
    vectors_batch = []

    for i, paper in enumerate(papers):
        logger.info(f"  Storing [{i+1}/{len(papers)}]: {paper.title[:50]}")

        text = paper_to_text(paper)
        embedding = get_embedding(openai_client, text)
        metadata = paper_to_metadata(paper)

        vectors_batch.append({
            "id": paper.paper_id,
            "values": embedding,
            "metadata": metadata,
        })

        stored.append(paper)

        # Upsert in batches
        if len(vectors_batch) >= BATCH_SIZE:
            index.upsert(vectors=vectors_batch)
            logger.info(f"  Upserted batch of {len(vectors_batch)} vectors")
            vectors_batch = []
            time.sleep(0.5)

    # Upsert remaining
    if vectors_batch:
        index.upsert(vectors=vectors_batch)
        logger.info(f"  Upserted final batch of {len(vectors_batch)} vectors")

    state["stored_papers"] = stored
    logger.info(f"Store node complete: {len(stored)} papers stored in Pinecone")
    return state
