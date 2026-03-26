# Research Paper ETL Pipeline

A full-stack AI-powered ETL pipeline using **LangGraph** agents to automatically traverse citation chains from a single seed paper, extract structured metadata via LLM, and build deduplicated research datasets stored in Pinecone.

Built for researchers and analysts who need to quickly map the citation landscape around a paper without manually opening and cataloging each reference.

## How It Works

Given a single seed paper URL, the pipeline:

1. **Fetches** the paper and its references from Semantic Scholar (with Pinecone pre-check to skip known papers)
2. **Parses** raw API responses into structured records
3. **Extracts** and validates metadata via a local LLM (Qwen 2.5)
4. **Resolves** citations recursively up to N hops deep
5. **Reviews** low-confidence records with human-in-the-loop interrupt
6. **Deduplicates** against Pinecone to prevent cyclic traversal
7. **Stores** final records in Pinecone with vector embeddings

## Architecture

The pipeline is modeled as a **stateful directed graph** using LangGraph with 7 sequential nodes:

```
seed_url
   │
   ▼
[FETCH] ──▶ [PARSE] ──▶ [EXTRACT] ──▶ [RESOLVE] ──▶ [REVIEW] ──▶ [DEDUPLICATE] ──▶ [STORE]
   ▲                                      │                                            │
   │                                      │                                            │
   └──────── next citation hop ───────────┘                                            │
                                                                                       ▼
                                                                              Pinecone Index
```

Each node's output feeds directly into the next. State is persisted across all nodes via LangGraph's checkpointing, enabling pause/resume and crash recovery.

## Demo Results

Using "Attention Is All You Need" (Vaswani et al., 2017) as the seed paper:

```
RESULTS SUMMARY
════════════════════════════════════════════════════════════
Total papers fetched:      37
Papers stored in Pinecone: 36
Duplicates caught:         39 (on re-run via Pinecone pre-check)
Low confidence flagged:    0
Errors:                    0
Total time:                ~9 min (first run) / ~13s (cached re-run)
```

On re-run, the pipeline queries Pinecone **before** fetching — all 39 previously stored papers are skipped instantly, completing in 13 seconds with zero redundant API calls.

## Tech Stack

- **Orchestration**: LangGraph (stateful directed graph with 7 nodes + conditional edges)
- **LLM**: Qwen 2.5 (local, free) for metadata extraction and validation
- **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2) for vector generation
- **Vector Store**: Pinecone for deduplication, cycle detection, and persistent storage
- **Data Source**: Semantic Scholar API (free, no key required)
- **API**: FastAPI endpoint for triggering pipeline runs
- **State Management**: LangGraph persistence with checkpoint/resume support

## Quick Start

```bash
git clone https://github.com/Preksha2/research-paper-etl-pipeline.git
cd research-paper-etl-pipeline
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env      # Add your Pinecone API key
python scripts/demo.py
```

The demo uses "Attention Is All You Need" as the seed paper and traverses 1 hop of its citation chain.

To customize:
```bash
# Full 2-hop traversal
python scripts/demo.py --hops 2

# Different seed paper
python scripts/demo.py --url "https://arxiv.org/abs/2005.14165" --hops 1

# Use ArXiv ID directly
python scripts/demo.py --url "ArXiv:1810.04805" --hops 1
```

## Key Design Decisions

**Pinecone pre-check before fetching**: The fetch node queries Pinecone in batch before making any Semantic Scholar API calls. On a re-run of 39 papers, this reduced execution time from ~15 minutes to 13 seconds — zero redundant API calls, zero wasted compute.

**Stateful directed graph over linear scripts**: Each node is independently testable. The graph handles branching (continue traversal vs. proceed to review) via conditional edges, eliminating hand-coded glue scripts between steps.

**Exponential backoff with retry cap**: Rate-limited requests use progressive delays (1.5s → 2.5s → 4.5s) with a max of 3 retries per paper, preventing infinite retry loops while respecting API limits.

**Human-in-the-loop interrupt**: Low-confidence records (missing DOIs, ambiguous metadata) are flagged for review. LangGraph's checkpointing enables the pipeline to resume from the exact paused node after correction.

**Heuristic fallback for LLM extraction**: When the LLM returns invalid JSON or fails, a heuristic scorer (based on presence of DOI, year, abstract, authors) provides confidence scores — ensuring the pipeline never stalls on a single bad extraction.

**Shared traversal state**: All 7 nodes read/write from a single state dict persisted via LangGraph's memory layer. The pipeline can be killed and restarted mid-run with zero reprocessing of already-completed nodes.

## Project Structure

```
src/
├── nodes/                  # LangGraph node implementations
│   ├── fetch.py            # Semantic Scholar fetching + Pinecone pre-check
│   ├── parse.py            # Raw response parsing into Paper objects
│   ├── extract.py          # LLM metadata extraction + heuristic fallback
│   ├── resolve.py          # Citation chain resolution + hop management
│   ├── review.py           # Human-in-the-loop review + auto-fix
│   ├── deduplicate.py      # Pinecone deduplication + cycle detection
│   └── store.py            # Vector embedding + Pinecone storage
├── graph/                  # LangGraph pipeline definition
│   └── pipeline.py         # 7-node graph with conditional traversal edges
├── models/                 # Pydantic data models
│   └── paper.py            # Paper, Author, PipelineState
├── api/                    # FastAPI server
│   ├── server.py           # REST endpoints (/run, /health)
│   └── schemas.py          # Request/response models
└── utils/                  # Configuration and helpers
    └── config.py           # Config + env loader
scripts/
└── demo.py                 # End-to-end demo
tests/
├── test_fetch.py           # Paper ID extraction tests
├── test_parse.py           # Raw response parsing tests
├── test_extract.py         # Heuristic enrichment tests
├── test_review.py          # Auto-fix logic tests
├── test_models.py          # Data model tests
└── test_api.py             # FastAPI endpoint tests
configs/
└── config.yaml             # Pipeline configuration
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/run` | Start pipeline with a seed paper URL |
| GET | `/health` | Health check + service connectivity |

### Example

```bash
# Start the server
uvicorn src.api.server:app --reload --port 8000

# Trigger a pipeline run
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"seed_paper_url": "https://arxiv.org/abs/1706.03762", "max_hops": 2}'
```

## Configuration

Edit `configs/config.yaml` to customize:

- `max_citation_hops`: How deep to traverse (default: 2)
- `max_papers_per_hop`: Cap on papers fetched per hop (default: 50)
- `confidence_threshold`: Below this triggers human review (default: 0.7)
- `rate_limit_delay`: Seconds between Semantic Scholar API calls (default: 1.0)
- Pinecone index settings (dimension, metric, cloud region)

## Testing

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html
```

## License

MIT
