"""
Resolve node: handles citation chain traversal.
Collects reference IDs from current batch of papers and queues them
for the next hop, checking against already-processed papers to prevent cycles.
"""
from typing import List, Set
from loguru import logger

from src.models.paper import Paper
from src.utils.config import load_config

config = load_config()
PIPELINE_CONFIG = config["pipeline"]
MAX_HOPS = PIPELINE_CONFIG["max_citation_hops"]
MAX_PAPERS_PER_HOP = PIPELINE_CONFIG["max_papers_per_hop"]


def resolve_node(state: dict) -> dict:
    """
    LangGraph node: Resolve citations for the next traversal hop.
    
    Collects all reference IDs from extracted papers, filters out
    already-processed papers, and queues new papers for fetching.
    Also updates the hop counter.
    """
    extracted = state.get("extracted_papers", [])
    current_hop = state.get("current_hop", 0)
    max_hops = state.get("max_hops", MAX_HOPS)
    already_processed = set(state.get("all_processed_ids", []))

    # Collect all reference IDs from current batch
    all_refs: Set[str] = set()
    for paper in extracted:
        for ref_id in paper.references:
            if ref_id and ref_id not in already_processed:
                all_refs.add(ref_id)

    # Mark current papers as processed
    for paper in extracted:
        already_processed.add(paper.paper_id)

    # Limit papers per hop
    new_refs = list(all_refs)[:MAX_PAPERS_PER_HOP]

    logger.info(
        f"Resolve node: hop {current_hop + 1}/{max_hops} | "
        f"{len(all_refs)} unique refs found | "
        f"{len(new_refs)} queued (max {MAX_PAPERS_PER_HOP}) | "
        f"{len(already_processed)} total processed"
    )

    state["papers_to_fetch"] = new_refs
    state["all_processed_ids"] = list(already_processed)
    state["current_hop"] = current_hop + 1

    return state
