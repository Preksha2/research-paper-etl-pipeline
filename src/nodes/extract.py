"""
Extract node: uses OpenAI LLM to enrich and validate paper metadata.
Handles missing DOIs, incomplete author lists, and abstract summarization.
Flags low-confidence records for human review.
"""
import os
import json
from typing import List
from loguru import logger
from openai import OpenAI

from src.models.paper import Paper
from src.utils.config import load_config

config = load_config()
LLM_CONFIG = config["openai"]
CONFIDENCE_THRESHOLD = config["pipeline"]["confidence_threshold"]


def get_openai_client() -> OpenAI:
    """Initialize OpenAI client."""
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def enrich_paper_with_llm(client: OpenAI, paper: Paper) -> Paper:
    """
    Use LLM to validate and enrich a single paper's metadata.
    
    Checks for:
        - Missing or malformed DOIs
        - Incomplete author lists
        - Missing year or abstract
        - Confidence scoring
    """
    prompt = f"""You are a research paper metadata validator. Given the following paper metadata, 
check for completeness and correctness. Return a JSON object with these fields:

- "doi": the DOI if you can identify it (or null)
- "year": publication year (or null)
- "confidence": float 0-1 indicating how complete/reliable the metadata is
- "notes": any issues found (string)

Paper:
- Title: {paper.title}
- Authors: {', '.join(a.name for a in paper.authors)}
- Year: {paper.year}
- DOI: {paper.doi}
- Abstract: {(paper.abstract or 'N/A')[:300]}

Return ONLY valid JSON, no explanation."""

    try:
        response = client.chat.completions.create(
            model=LLM_CONFIG["model"],
            messages=[
                {"role": "system", "content": "You are a metadata validation assistant. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=LLM_CONFIG["temperature"],
            max_tokens=300,
        )

        content = response.choices[0].message.content.strip()
        # Clean potential markdown fences
        content = content.replace("`json", "").replace("`", "").strip()
        result = json.loads(content)

        # Update paper with LLM enrichment
        if result.get("doi") and not paper.doi:
            paper.doi = result["doi"]
        if result.get("year") and not paper.year:
            paper.year = result["year"]

        paper.confidence = result.get("confidence", paper.confidence)
        paper.extraction_notes = result.get("notes", "")

        return paper

    except json.JSONDecodeError as e:
        logger.warning(f"LLM returned invalid JSON for '{paper.title[:40]}': {e}")
        paper.confidence = 0.5
        paper.extraction_notes = "LLM extraction failed - invalid JSON response"
        return paper

    except Exception as e:
        logger.error(f"LLM extraction error for '{paper.title[:40]}': {e}")
        paper.confidence = 0.5
        paper.extraction_notes = f"LLM extraction error: {str(e)}"
        return paper


def extract_node(state: dict) -> dict:
    """
    LangGraph node: Enrich parsed papers using OpenAI LLM.
    
    Reads from state['parsed_papers'], enriches each via LLM,
    flags low-confidence records for human review.
    """
    parsed = state.get("parsed_papers", [])

    if not parsed:
        logger.warning("Extract node: no papers to extract")
        state["extracted_papers"] = []
        return state

    client = get_openai_client()
    extracted = []
    needs_review = []

    for i, paper in enumerate(parsed):
        logger.info(f"  Extracting [{i+1}/{len(parsed)}]: {paper.title[:60]}")
        enriched = enrich_paper_with_llm(client, paper)
        extracted.append(enriched)

        if enriched.confidence < CONFIDENCE_THRESHOLD:
            needs_review.append(enriched)
            logger.warning(f"    Low confidence ({enriched.confidence:.2f}): flagged for review")

    state["extracted_papers"] = extracted
    state["needs_human_review"] = needs_review
    state["low_confidence_count"] = state.get("low_confidence_count", 0) + len(needs_review)

    logger.info(f"Extract node complete: {len(extracted)} enriched, {len(needs_review)} flagged for review")
    return state
