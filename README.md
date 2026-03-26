# Research Paper ETL Pipeline

A full-stack AI-powered ETL pipeline using **LangGraph** agents to automatically traverse citation chains from a single seed paper, extract structured metadata via LLM, and build deduplicated research datasets stored in Pinecone.

Built for researchers and analysts who need to quickly map the citation landscape around a paper without manually opening and cataloging each reference.

## How It Works

Given a single seed paper URL, the pipeline:

1. **Fetches** the paper and its references from Semantic Scholar
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

## Tech Stack

- **Orchestration**: LangGraph (stateful directed graph with 7 nodes)
- **LLM**: Qwen 2.5 (local, free) for metadata extraction — OpenAI supported as optional backend
- **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2) for vector generation
- **Vector Store**: Pinecone for deduplication and persistent storage
- **Data Source**: Semantic Scholar API (free, no key required)
- **API**: FastAPI endpoint for triggering pipeline runs
- **State Management**: LangGraph persistence with checkpoint/resume

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
# Different seed paper
python scripts/demo.py --url "https://arxiv.org/abs/2005.14165" --hops 2

# Use ArXiv ID directly
python scripts/demo.py --url "ArXiv:1810.04805" --hops 1
```

## Demo Output

```
RESULTS SUMMARY
════════════════════════════════════════════════════════════
Total papers fetched:      1
Papers stored in Pinecone: 1
Duplicates caught:         0
Low confidence flagged:    0
Errors:                    0
Total time:                58.7s

Papers stored:
────────────────────────────────────────────────────────────
  1. Attention is All you Need
     Year: 2017 | DOI: N/A | Confidence: 0.80
     Authors: Ashish Vaswani, Noam Shazeer, Niki Parmar
```

## Project Structure

```
src/
├── nodes/                  # LangGraph node implementations
│   ├── fetch.py            # Paper fetching from Semantic Scholar
│   ├── parse.py            # Raw response parsing into Paper objects
│   ├── extract.py          # LLM-powered metadata extraction + validation
│   ├── resolve.py          # Citation chain resolution + hop management
│   ├── review.py           # Human-in-the-loop review + auto-fix
│   ├── deduplicate.py      # Pinecone-based deduplication + cycle detection
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
  -d '{"seed_paper_url": "https://arxiv.org/abs/1706.03762", "max_hops": 1}'
```

## Key Design Decisions

**Stateful directed graph over linear scripts**: Each node is independently testable and the graph handles branching (continue traversal vs. proceed to review) via conditional edges, eliminating hand-coded glue scripts.

**Conditional citation resolution**: The resolve node queries Pinecone before fetching, skipping already-processed papers and preventing cyclic loops in deeply nested citation networks.

**Human-in-the-loop interrupt**: Low-confidence records (missing DOIs, ambiguous metadata) are flagged for review. LangGraph's checkpointing enables the pipeline to resume from the exact paused node after correction.

**Shared traversal state**: All 7 nodes read/write from a single `PipelineState` object persisted via LangGraph's memory layer, validated by the ability to kill and restart the pipeline mid-run with zero reprocessing.

**Local-first LLM**: Uses Qwen 2.5 (0.5B) locally for metadata extraction — no API keys or costs for the LLM component. Pinecone free tier handles vector storage.

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
