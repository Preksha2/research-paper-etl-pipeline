"""
Extract node: uses a local HuggingFace LLM to enrich and validate paper metadata.
Handles missing DOIs, incomplete author lists, and abstract summarization.
Flags low-confidence records for human review.
"""
import json
import re
from typing import Optional
from loguru import logger

from src.models.paper import Paper
from src.utils.config import load_config

config = load_config()
CONFIDENCE_THRESHOLD = config["pipeline"]["confidence_threshold"]

# Lazy-loaded LLM pipeline
_llm_pipe = None


def get_llm_pipeline():
    """Initialize local LLM pipeline (lazy loaded, singleton)."""
    global _llm_pipe
    if _llm_pipe is None:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        logger.info(f"Loading local LLM: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        _llm_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=300,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        logger.info(f"Local LLM loaded: {model_name}")
    return _llm_pipe


def enrich_paper_with_llm(paper: Paper) -> Paper:
    """
    Use local LLM to validate and enrich a single paper's metadata.
    """
    pipe = get_llm_pipeline()

    prompt_messages = [
        {"role": "system", "content": "You are a research paper metadata validator. Return only valid JSON with fields: doi (string or null), year (int or null), confidence (float 0-1), notes (string)."},
        {"role": "user", "content": f"Validate this paper metadata and return JSON:\nTitle: {paper.title}\nAuthors: {', '.join(a.name for a in paper.authors[:5])}\nYear: {paper.year}\nDOI: {paper.doi}\nAbstract: {(paper.abstract or 'N/A')[:200]}"},
    ]

    try:
        tokenizer = pipe.tokenizer
        prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

        output = pipe(prompt)
        generated = output[0]["generated_text"]

        if generated.startswith(prompt):
            response_text = generated[len(prompt):].strip()
        else:
            response_text = generated.strip()

        json_match = re.search(r'\{[^{}]+\}', response_text)
        if json_match:
            result = json.loads(json_match.group())

            if result.get("doi") and not paper.doi:
                paper.doi = result["doi"]
            if result.get("year") and not paper.year:
                paper.year = result["year"]

            conf = result.get("confidence")
            if conf is not None:
                paper.confidence = float(conf)
            paper.extraction_notes = result.get("notes", "")
        else:
            paper = _heuristic_enrich(paper)

    except Exception as e:
        logger.warning(f"LLM extraction failed for '{paper.title[:40]}': {e}")
        paper = _heuristic_enrich(paper)

    # Ensure confidence is never None
    if paper.confidence is None:
        paper = _heuristic_enrich(paper)

    return paper


def _heuristic_enrich(paper: Paper) -> Paper:
    """Fallback heuristic enrichment when LLM fails."""
    score = 0.5

    if paper.doi:
        score += 0.2
    if paper.year:
        score += 0.1
    if paper.abstract and len(paper.abstract) > 100:
        score += 0.1
    if len(paper.authors) > 0:
        score += 0.1

    paper.confidence = min(score, 1.0)
    paper.extraction_notes = "Enriched via heuristic fallback"
    return paper


def extract_node(state: dict) -> dict:
    """
    LangGraph node: Enrich parsed papers using local LLM.
    """
    parsed = state.get("parsed_papers", [])

    if not parsed:
        logger.warning("Extract node: no papers to extract")
        state["extracted_papers"] = []
        return state

    extracted = []
    needs_review = []

    for i, paper in enumerate(parsed):
        logger.info(f"  Extracting [{i+1}/{len(parsed)}]: {paper.title[:60]}")
        enriched = enrich_paper_with_llm(paper)
        extracted.append(enriched)

        if enriched.confidence < CONFIDENCE_THRESHOLD:
            needs_review.append(enriched)
            logger.warning(f"    Low confidence ({enriched.confidence:.2f}): flagged for review")

    state["extracted_papers"] = extracted
    state["needs_human_review"] = needs_review
    state["low_confidence_count"] = state.get("low_confidence_count", 0) + len(needs_review)

    logger.info(f"Extract node complete: {len(extracted)} enriched, {len(needs_review)} flagged for review")
    return state
