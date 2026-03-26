# Research Paper ETL Pipeline

A full-stack AI-powered ETL pipeline using **LangGraph** agents to automatically traverse citation chains from a single seed paper, extract structured metadata via LLM, and build deduplicated research datasets stored in Pinecone.

Built for researchers and analysts who need to quickly map the citation landscape around a paper without manually opening and cataloging each reference.

## How It Works

Given a single seed paper URL, the pipeline:

1. **Fetches** the paper and its references from Semantic Scholar
2. **Parses** raw API responses into structured records
3. **Extracts** metadata (authors, DOI, year, abstract) via OpenAI LLM
4. **Resolves** citations recursively up to N hops deep
5. **Reviews** low-confidence records with human-in-the-loop interrupt
6. **Deduplicates** against Pinecone to prevent cyclic traversal
7. **Stores** final records in Pinecone with vector embeddings

## Architecture

The pipeline is modeled as a **stateful directed graph** using LangGraph with 7 sequential nodes:

`
seed_url
   |
   v
[FETCH] --> [PARSE] --> [EXTRACT] --> [RESOLVE] --> [REVIEW] --> [DEDUPLICATE] --> [STORE]
   ^                                     |                                           |
   |                                     |                                           |
   +--------- next citation hop ---------+                                           |
                                                                                     v
                                                                            Pinecone Index
`

Each node's output feeds directly into the next. State is persisted across all nodes via LangGraph's checkpointing, enabling pause/resume and crash recovery.

## Tech Stack

- **Orchestration**: LangGraph (stateful directed graph with 7 nodes)
- **LLM**: OpenAI GPT-3.5-turbo for metadata extraction
- **Vector Store**: Pinecone for deduplication and storage
- **Data Source**: Semantic Scholar API (free, no key required)
- **API**: FastAPI endpoint for triggering pipeline runs
- **State Management**: LangGraph persistence with checkpoint/resume

## Quick Start

`ash
git clone https://github.com/Preksha2/research-paper-etl-pipeline.git
cd research-paper-etl-pipeline
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env      # Add your OpenAI and Pinecone API keys
python scripts/demo.py
`

## Project Structure

`
src/
+-- nodes/              # LangGraph node implementations
|   +-- fetch.py        # Paper fetching from Semantic Scholar
|   +-- parse.py        # Raw response parsing
|   +-- extract.py      # LLM-powered metadata extraction
|   +-- resolve.py      # Citation chain resolution
|   +-- review.py       # Human-in-the-loop review
|   +-- deduplicate.py  # Pinecone-based deduplication
|   +-- store.py        # Vector storage in Pinecone
+-- graph/              # LangGraph pipeline definition
|   +-- pipeline.py     # Graph construction and execution
+-- models/             # Pydantic data models
|   +-- paper.py        # Paper, Author, PipelineState
+-- api/                # FastAPI server
|   +-- server.py       # REST endpoint
+-- utils/              # Configuration and helpers
    +-- config.py       # Config loader
scripts/
+-- demo.py             # End-to-end demo
configs/
+-- config.yaml         # Pipeline configuration
`

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /run | Start pipeline with a seed paper URL |
| GET | /status/{run_id} | Check pipeline run status |
| GET | /health | Health check |

## Configuration

Edit configs/config.yaml to customize:
- Maximum citation hops (default: 2)
- Papers per hop limit
- LLM confidence threshold for human review
- Pinecone index settings
- Rate limiting for Semantic Scholar API

## License

MIT
