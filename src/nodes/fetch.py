"""
Fetch node: retrieves paper data from Semantic Scholar API.
Checks Pinecone before fetching to skip already-processed papers,
preventing unnecessary API calls and cyclic traversal.
"""
import os
import re
import time
from typing import Dict, Any
import httpx
from loguru import logger

from src.utils.config import load_config

config = load_config()
SS_CONFIG = config["semantic_scholar"]
BASE_URL = SS_CONFIG["base_url"]
FIELDS = SS_CONFIG["fields"]
RATE_LIMIT = SS_CONFIG["rate_limit_delay"]


def extract_paper_id(url_or_id: str) -> str:
    """
    Extract Semantic Scholar paper ID from various URL formats.
    """
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


def fetch_paper(paper_id: str) -> Dict[str, Any]:
    """Fetch a single paper from Semantic Scholar API."""
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
        elif e.response.status_code == 429:
            logger.warning(f"Rate limited, waiting 5s...")
            time.sleep(5)
            return fetch_paper(paper_id)
        else:
            logger.error(f"HTTP error fetching {paper_id}: {e.response.status_code}")
        return {}

    except Exception as e:
        logger.error(f"Error fetching {paper_id}: {e}")
        return {}


def fetch_node(state: dict) -> dict:
    """
    LangGraph node: Fetch papers from Semantic Scholar.
    Queries Pinecone before each fetch to skip already-processed papers.
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

    # Filter out already processed papers
    to_fetch = [pid for pid in papers_to_fetch if pid not in already_processed]

    # Pre-fetch Pinecone check: skip papers already in the index
    skipped_pinecone = 0
    final_fetch_list = []
    for pid in to_fetch:
        if check_pinecone_exists(pid):
            logger.info(f"  Skipped (already in Pinecone): {pid[:50]}")
            skipped_pinecone += 1
            already_processed.add(pid)
        else:
            final_fetch_list.append(pid)

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
