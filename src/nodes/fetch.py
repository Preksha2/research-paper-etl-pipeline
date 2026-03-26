"""
Fetch node: retrieves paper data from Semantic Scholar API.
Handles both seed paper URLs and paper IDs for citation traversal.
"""
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
    Supports:
        - Semantic Scholar URLs
        - ArXiv URLs/IDs
        - DOIs
        - Direct Semantic Scholar IDs
    """
    # Semantic Scholar URL
    match = re.search(r'semanticscholar\.org/paper/[^/]*?/([a-f0-9]{40})', url_or_id)
    if match:
        return match.group(1)

    # Semantic Scholar URL (alternate format)
    match = re.search(r'semanticscholar\.org/paper/([a-f0-9]{40})', url_or_id)
    if match:
        return match.group(1)

    # ArXiv URL
    match = re.search(r'arxiv\.org/abs/(\d+\.\d+)', url_or_id)
    if match:
        return f"ArXiv:{match.group(1)}"

    # DOI
    if url_or_id.startswith("10."):
        return f"DOI:{url_or_id}"

    # DOI URL
    match = re.search(r'doi\.org/(10\..+)', url_or_id)
    if match:
        return f"DOI:{match.group(1)}"

    # Already a Semantic Scholar ID or ArXiv ID
    return url_or_id


def fetch_paper(paper_id: str) -> Dict[str, Any]:
    """
    Fetch a single paper from Semantic Scholar API.

    Args:
        paper_id: Semantic Scholar paper ID, ArXiv ID, or DOI.

    Returns:
        Raw API response as a dict.
    """
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

    Reads paper IDs from state['papers_to_fetch'],
    fetches each one, and stores raw responses in state['fetched_raw'].
    """
    papers_to_fetch = state.get("papers_to_fetch", [])
    already_processed = set(state.get("all_processed_ids", []))

    if not papers_to_fetch:
        # First run: fetch seed paper
        seed_url = state.get("seed_paper_url", "")
        if not seed_url:
            logger.error("No seed paper URL provided")
            return state

        paper_id = extract_paper_id(seed_url)
        papers_to_fetch = [paper_id]

    # Filter out already processed papers
    to_fetch = [pid for pid in papers_to_fetch if pid not in already_processed]
    logger.info(f"Fetch node: {len(to_fetch)} papers to fetch ({len(papers_to_fetch) - len(to_fetch)} skipped as already processed)")

    fetched = []
    for i, paper_id in enumerate(to_fetch):
        logger.info(f"  Fetching [{i+1}/{len(to_fetch)}]: {paper_id[:60]}")
        data = fetch_paper(paper_id)
        if data:
            fetched.append(data)
        time.sleep(RATE_LIMIT)

    state["fetched_raw"] = fetched
    state["total_fetched"] = state.get("total_fetched", 0) + len(fetched)
    logger.info(f"Fetch node complete: {len(fetched)} papers fetched")
    return state
