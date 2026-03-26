"""
End-to-end demo of the Research Paper ETL Pipeline.
Runs the full LangGraph pipeline from a seed paper through all 7 nodes.

Usage:
    python scripts/demo.py
    python scripts/demo.py --url "https://arxiv.org/abs/1706.03762"
    python scripts/demo.py --hops 1
"""
import os
import sys
import json
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from src.graph.pipeline import run_pipeline
from src.utils.config import load_env


# "Attention Is All You Need" - the foundational transformer paper
DEFAULT_SEED = "https://arxiv.org/abs/1706.03762"


def run_demo(seed_url: str, max_hops: int = 1):
    """Run the full pipeline demo."""

    load_env()

    logger.info("=" * 60)
    logger.info("RESEARCH PAPER ETL PIPELINE - DEMO")
    logger.info("=" * 60)
    logger.info(f"Seed paper: {seed_url}")
    logger.info(f"Max hops: {max_hops}")
    logger.info("")

    start_time = time.time()

    # Run the pipeline
    final_state = run_pipeline(
        seed_paper_url=seed_url,
        max_hops=max_hops,
        thread_id="demo-run",
    )

    elapsed = time.time() - start_time

    # Print results
    stored = final_state.get("stored_papers", [])
    logger.info("")
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total papers fetched:     {final_state.get('total_fetched', 0)}")
    logger.info(f"Papers stored in Pinecone: {len(stored)}")
    logger.info(f"Duplicates caught:        {final_state.get('duplicate_count', 0)}")
    logger.info(f"Low confidence flagged:   {final_state.get('low_confidence_count', 0)}")
    logger.info(f"Errors:                   {len(final_state.get('errors', []))}")
    logger.info(f"Total time:               {elapsed:.1f}s")
    logger.info("")

    if stored:
        logger.info("Papers stored:")
        logger.info("-" * 80)
        for i, paper in enumerate(stored, 1):
            title = paper.title if hasattr(paper, 'title') else paper.get('title', 'Unknown')
            year = paper.year if hasattr(paper, 'year') else paper.get('year', 'N/A')
            doi = paper.doi if hasattr(paper, 'doi') else paper.get('doi', 'N/A')
            conf = paper.confidence if hasattr(paper, 'confidence') else paper.get('confidence', 0)
            authors = ", ".join(a.name for a in paper.authors[:3]) if hasattr(paper, 'authors') else str(paper.get('authors', ''))[:60]

            logger.info(f"  {i:3d}. {title[:70]}")
            logger.info(f"       Year: {year} | DOI: {doi or 'N/A'} | Confidence: {conf:.2f}")
            logger.info(f"       Authors: {authors}")
            logger.info("")

    # Save results to file
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "demo_results.json")

    results = {
        "seed_paper": seed_url,
        "max_hops": max_hops,
        "total_fetched": final_state.get("total_fetched", 0),
        "papers_stored": len(stored),
        "duplicates_caught": final_state.get("duplicate_count", 0),
        "low_confidence_flagged": final_state.get("low_confidence_count", 0),
        "elapsed_seconds": round(elapsed, 1),
        "papers": [
            {
                "paper_id": p.paper_id if hasattr(p, 'paper_id') else p.get('paper_id', ''),
                "title": p.title if hasattr(p, 'title') else p.get('title', ''),
                "year": p.year if hasattr(p, 'year') else p.get('year'),
                "doi": p.doi if hasattr(p, 'doi') else p.get('doi'),
                "confidence": p.confidence if hasattr(p, 'confidence') else p.get('confidence', 0),
            }
            for p in stored
        ],
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Research Paper ETL Pipeline demo")
    parser.add_argument("--url", type=str, default=DEFAULT_SEED,
                        help=f"Seed paper URL or ID (default: {DEFAULT_SEED})")
    parser.add_argument("--hops", type=int, default=1,
                        help="Max citation hops (default: 1)")
    args = parser.parse_args()
    run_demo(seed_url=args.url, max_hops=args.hops)
