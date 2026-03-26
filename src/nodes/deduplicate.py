"""
Deduplicate node: checks papers against Pinecone index to prevent
storing duplicates and catching cyclic references in citation chains.
"""
import os
from typing import List, Tuple
from loguru import logger
from pinecone import Pinecone
from openai import OpenAI

from src.models.paper import Paper
from src.utils.config import load_config

config = load_config()
PINECONE_CONFIG = config["pinecone"]


def get_pinecone_client() -> Pinecone:
    """Initialize Pinecone client."""
    return Pinecone(api_key=os.getenv("PINECONE_API_KEY"))


def get_embedding(client: OpenAI, text: str) -> List[float]:
    """Generate embedding for a text using OpenAI."""
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text[:8000],
    )
    return response.data[0].embedding


def ensure_index_exists(pc: Pinecone, index_name: str) -> None:
    """Create Pinecone index if it doesn't exist."""
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        from pinecone import ServerlessSpec
        pc.create_index(
            name=index_name,
            dimension=PINECONE_CONFIG["dimension"],
            metric=PINECONE_CONFIG["metric"],
            spec=ServerlessSpec(
                cloud=PINECONE_CONFIG["cloud"],
                region=PINECONE_CONFIG["region"],
            ),
        )
        logger.info(f"Created Pinecone index: {index_name}")
        # Wait for index to be ready
        import time
        time.sleep(10)
    else:
        logger.info(f"Pinecone index '{index_name}' already exists")


def check_duplicate(index, paper_id: str) -> bool:
    """Check if a paper already exists in Pinecone by ID."""
    try:
        result = index.fetch(ids=[paper_id])
        return paper_id in result.get("vectors", {})
    except Exception:
        return False


def deduplicate_node(state: dict) -> dict:
    """
    LangGraph node: Deduplicate papers against Pinecone.

    Queries Pinecone for each paper ID before allowing it through.
    Catches duplicate entries and prevents cyclic loops in
    deeply nested citation networks.
    """
    reviewed = state.get("reviewed_papers", [])

    if not reviewed:
        logger.warning("Deduplicate node: no papers to deduplicate")
        state["deduplicated_papers"] = []
        return state

    pc = get_pinecone_client()
    index_name = os.getenv("PINECONE_INDEX_NAME", PINECONE_CONFIG["index_name"])
    ensure_index_exists(pc, index_name)
    index = pc.Index(index_name)

    unique = []
    duplicates = 0

    for paper in reviewed:
        if check_duplicate(index, paper.paper_id):
            logger.info(f"  Duplicate skipped: {paper.title[:50]}")
            duplicates += 1
        else:
            unique.append(paper)

    state["deduplicated_papers"] = unique
    state["duplicate_count"] = state.get("duplicate_count", 0) + duplicates

    logger.info(f"Deduplicate node complete: {len(unique)} unique, {duplicates} duplicates caught")
    return state
