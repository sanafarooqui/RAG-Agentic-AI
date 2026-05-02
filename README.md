# RAG-Agentic-AI

A collection of projects exploring Retrieval-Augmented Generation (RAG) and Agentic AI patterns. Each sub-project is self-contained and demonstrates a different technique or use case.

## Tech Stack

- **Language**: Python 3.11+
- **LLM Orchestration**: LangChain, LangGraph
- **Agent Frameworks**: smolagents
- **LLM Providers**: OpenAI (gpt-4o-mini), compatible with any OpenAI-API-compatible endpoint
- **Vector Store**: ChromaDB
- **Embeddings**: OpenAI text-embedding-ada-002

## Projects

### [`RAG-research-papers-assistant`](RAG-research-papers-assistant/)

A RAG pipeline that ingests a corpus of agentic AI research papers from a zip archive, stores them in a Chroma vector store, and answers questions grounded in the papers.

Includes an LLM-as-a-judge evaluation module that scores responses on **groundedness** (no hallucination) and **relevance** (answers the question).

| File | Purpose |
|---|---|
| `main.py` | Ingest pipeline + query interface |
| `evaluate.py` | LLM-as-a-judge evaluation (groundedness & relevance) |

**Quickstart**

```bash
# Activate the shared venv
source .venv/bin/activate

# 1. Ingest PDFs into the vector store (run once)
cd RAG-research-papers-assistant
python main.py ingest

# 2. Query
python main.py "What are the key challenges in agentic AI?"

# 3. Evaluate RAG quality
python evaluate.py
```

**Environment** — create `RAG-research-papers-assistant/.env`:
```
OPENAI_API_KEY=your-key
OPENAI_API_BASE=https://api.openai.com/v1
```

## Repository Structure

```
RAG-Agentic-AI/
├── .venv/                          # Shared virtual environment
├── RAG-research-papers-assistant/
│   ├── main.py                     # RAG pipeline
│   ├── evaluate.py                 # LLM-as-a-judge evaluation
│   ├── agentic_ai_research_papers.zip
│   └── .env                        # API keys (not committed)
└── README.md
```

## Setup

```bash
# Clone and create the shared venv
git clone <repo-url>
cd RAG-Agentic-AI
python3 -m venv .venv
source .venv/bin/activate
pip install langchain langchain-community langchain-chroma langchain-openai \
            langchain-text-splitters chromadb pypdf tiktoken pandas python-dotenv
```
