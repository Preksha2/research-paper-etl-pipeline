"""
Fetch node: retrieves paper data from Semantic Scholar API.
Checks Pinecone before fetching to skip already-processed papers.
Uses exponential backoff and reduced wait times for rate limiting.
"""
import os
import re
import time
from typing import Dict, Any, List
import httpx
from loguru import logger

from src.utils.config import load_config

config = load_config()
SS_CONFIG = config["semantic_scholar"]
BASE_URL = SS_CONFIG["base_url"]
FIELDS = SS_CONFIG["fields"]
RATE_LIMIT = SS_CONFIG["rate_limit_delay"]


def extract_paper_id(url_or_id: str) -> str:
    """Extract Semantic Scholar paper ID from various URL formats."""
    match = re.search(r'semanticscholar\.org/paper/[^/]*?/([a-f0-9]{40})', url_or_id)
    if match:
        return match.group(1)

    match = re.search(r'semanticscholar\.org/paper/([a-f0-9]{40})', url_or_id)
    if match:
        return match.group(1)

    match = re.search(r'arxiv\.org/abs/(\d+\.\d+)', url_or_id)
    if match:
        return f"ArXiv:{match.group(1)}"

    if url_or_id.startswith("10."):
        return f"DOI:{url_or_id}"

    match = re.search(r'doi\.org/(10\..+)', url_or_id)
    if match:
        return f"DOI:{match.group(1)}"

    return url_or_id


def check_pinecone_exists(paper_id: str) -> bool:
    """Check if a paper already exists in Pinecone before fetching."""
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = os.getenv("PINECONE_INDEX_NAME", "citation-index")

        existing = [idx.name for idx in pc.list_indexes()]
        if index_name not in existing:
            return False

        index = pc.Index(index_name)
        result = index.fetch(ids=[paper_id])
        return paper_id in result.get("vectors", {})
    except Exception:
        return False


def check_pinecone_batch(paper_ids: List[str]) -> set:
    """Batch check which papers already exist in Pinecone."""
    existing_ids = set()
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = os.getenv("PINECONE_INDEX_NAME", "citation-index")

        existing_indexes = [idx.name for idx in pc.list_indexes()]
        if index_name not in existing_indexes:
            return existing_ids

        index = pc.Index(index_name)

        # Batch fetch in chunks of 100
        for i in range(0, len(paper_ids), 100):
            batch = paper_ids[i:i+100]
            result = index.fetch(ids=batch)
            existing_ids.update(result.get("vectors", {}).keys())

    except Exception as e:
        logger.warning(f"Pinecone batch check failed: {e}")

    return existing_ids


def fetch_paper(paper_id: str, retry_count: int = 0, max_retries: int = 3) -> Dict[str, Any]:
    """Fetch a single paper with exponential backoff."""
    url = f"{BASE_URL}/paper/{paper_id}"
    params = {"fields": FIELDS}

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            logger.info(f"Fetched: {data.get('title', 'Unknown')[:80]}")
            return data

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            logger.warning(f"Paper not found: {paper_id}")
            return {}
        elif e.response.status_code == 429:
            if retry_count >= max_retries:
                logger.error(f"Max retries reached for {paper_id}")
                return {}
            wait = min(2 ** retry_count + 0.5, 10)
            logger.debug(f"Rate limited, waiting {wait:.1f}s (retry {retry_count + 1}/{max_retries})")
            time.sleep(wait)
            return fetch_paper(paper_id, retry_count + 1, max_retries)
        else:
            logger.error(f"HTTP error fetching {paper_id}: {e.response.status_code}")
            return {}

    except Exception as e:
        logger.error(f"Error fetching {paper_id}: {e}")
        return {}


def fetch_node(state: dict) -> dict:
    """
    LangGraph node: Fetch papers from Semantic Scholar.
    Queries Pinecone in batch before fetching to skip already-processed papers.
    """
    papers_to_fetch = state.get("papers_to_fetch", [])
    already_processed = set(state.get("all_processed_ids", []))

    if not papers_to_fetch:
        seed_url = state.get("seed_paper_url", "")
        if not seed_url:
            logger.error("No seed paper URL provided")
            return state
        paper_id = extract_paper_id(seed_url)
        papers_to_fetch = [paper_id]

    # Filter out already processed papers (in-memory check)
    to_fetch = [pid for pid in papers_to_fetch if pid not in already_processed]

    # Batch Pinecone pre-check: skip papers already in the index
    if to_fetch:
        pinecone_existing = check_pinecone_batch(to_fetch)
        skipped_pinecone = len(pinecone_existing)
        already_processed.update(pinecone_existing)
        final_fetch_list = [pid for pid in to_fetch if pid not in pinecone_existing]
    else:
        skipped_pinecone = 0
        final_fetch_list = []

    logger.info(
        f"Fetch node: {len(final_fetch_list)} to fetch | "
        f"{len(papers_to_fetch) - len(to_fetch)} already processed | "
        f"{skipped_pinecone} skipped via Pinecone pre-check"
    )

    fetched = []
    for i, paper_id in enumerate(final_fetch_list):
        logger.info(f"  Fetching [{i+1}/{len(final_fetch_list)}]: {paper_id[:60]}")
        data = fetch_paper(paper_id)
        if data:
            fetched.append(data)
        time.sleep(RATE_LIMIT)

    state["fetched_raw"] = fetched
    state["all_processed_ids"] = list(already_processed)
    state["total_fetched"] = state.get("total_fetched", 0) + len(fetched)
    state["duplicate_count"] = state.get("duplicate_count", 0) + skipped_pinecone
    logger.info(f"Fetch node complete: {len(fetched)} papers fetched, {skipped_pinecone} pre-filtered by Pinecone")
    return state
