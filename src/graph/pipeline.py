"""
LangGraph pipeline: wires all 7 nodes into a stateful directed graph
with conditional edges for citation hop traversal.

Graph structure:
    fetch -> parse -> extract -> resolve -> review -> deduplicate -> store
                                   |                                  |
                                   +--- (if more hops) ---> fetch ----+
                                   |
                                   +--- (if done) ---> review --------+

State is persisted across all nodes via LangGraph's built-in checkpointing,
enabling pause/resume and crash recovery.
"""
import os
from typing import Any, Dict
from loguru import logger

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.nodes.fetch import fetch_node
from src.nodes.parse import parse_node
from src.nodes.extract import extract_node
from src.nodes.resolve import resolve_node
from src.nodes.review import review_node
from src.nodes.deduplicate import deduplicate_node
from src.nodes.store import store_node
from src.utils.config import load_config

config = load_config()
MAX_HOPS = config["pipeline"]["max_citation_hops"]


def should_continue_traversal(state: dict) -> str:
    """
    Conditional edge: decide whether to continue fetching
    more citation hops or proceed to review.

    Returns:
        'fetch' if more hops remain and there are papers to fetch.
        'review' if traversal is complete.
    """
    current_hop = state.get("current_hop", 0)
    max_hops = state.get("max_hops", MAX_HOPS)
    papers_to_fetch = state.get("papers_to_fetch", [])

    if current_hop < max_hops and len(papers_to_fetch) > 0:
        logger.info(f"Continuing to hop {current_hop + 1}/{max_hops} ({len(papers_to_fetch)} papers queued)")
        return "fetch"
    else:
        reason = "max hops reached" if current_hop >= max_hops else "no more papers to fetch"
        logger.info(f"Traversal complete: {reason}")
        return "review"


def build_pipeline() -> StateGraph:
    """
    Build the LangGraph pipeline with all 7 nodes and conditional edges.

    Returns:
        Compiled StateGraph ready for execution.
    """
    # Define the graph with dict state
    workflow = StateGraph(dict)

    # Add all 7 nodes
    workflow.add_node("fetch", fetch_node)
    workflow.add_node("parse", parse_node)
    workflow.add_node("extract", extract_node)
    workflow.add_node("resolve", resolve_node)
    workflow.add_node("review", review_node)
    workflow.add_node("deduplicate", deduplicate_node)
    workflow.add_node("store", store_node)

    # Define edges: linear flow with conditional branch at resolve
    workflow.set_entry_point("fetch")
    workflow.add_edge("fetch", "parse")
    workflow.add_edge("parse", "extract")
    workflow.add_edge("extract", "resolve")

    # Conditional: either loop back to fetch or proceed to review
    workflow.add_conditional_edges(
        "resolve",
        should_continue_traversal,
        {
            "fetch": "fetch",
            "review": "review",
        }
    )

    workflow.add_edge("review", "deduplicate")
    workflow.add_edge("deduplicate", "store")
    workflow.add_edge("store", END)

    return workflow


def run_pipeline(
    seed_paper_url: str,
    max_hops: int = None,
    thread_id: str = "default",
) -> Dict[str, Any]:
    """
    Execute the full citation traversal pipeline.

    Args:
        seed_paper_url: URL or ID of the seed paper.
        max_hops: Override max citation hops (default from config).
        thread_id: Thread ID for checkpointing.

    Returns:
        Final pipeline state with all processed papers.
    """
    from src.utils.config import load_env
    load_env()

    hops = max_hops if max_hops is not None else MAX_HOPS

    # Initial state
    initial_state = {
        "seed_paper_url": seed_paper_url,
        "current_hop": 0,
        "max_hops": hops,
        "papers_to_fetch": [],
        "fetched_raw": [],
        "parsed_papers": [],
        "extracted_papers": [],
        "reviewed_papers": [],
        "deduplicated_papers": [],
        "stored_papers": [],
        "all_processed_ids": [],
        "duplicate_count": 0,
        "low_confidence_count": 0,
        "total_fetched": 0,
        "errors": [],
        "needs_human_review": [],
        "human_reviewed": False,
    }

    # Build and compile graph with checkpointing
    workflow = build_pipeline()
    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)

    logger.info("=" * 60)
    logger.info(f"Starting pipeline: seed={seed_paper_url}")
    logger.info(f"Max hops: {hops}")
    logger.info("=" * 60)

    # Run the pipeline
    config = {"configurable": {"thread_id": thread_id}}
    final_state = None

    for step in app.stream(initial_state, config=config):
        # Each step is a dict with node_name -> state
        node_name = list(step.keys())[0]
        final_state = step[node_name]
        logger.info(f"Completed node: {node_name}")

    if final_state is None:
        final_state = initial_state

    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info(f"  Total papers fetched: {final_state.get('total_fetched', 0)}")
    logger.info(f"  Papers stored: {len(final_state.get('stored_papers', []))}")
    logger.info(f"  Duplicates caught: {final_state.get('duplicate_count', 0)}")
    logger.info(f"  Low confidence flagged: {final_state.get('low_confidence_count', 0)}")
    logger.info(f"  Errors: {len(final_state.get('errors', []))}")
    logger.info("=" * 60)

    return final_state
