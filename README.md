# RAG-Agentic-AI

A collection of projects exploring Retrieval-Augmented Generation (RAG) and Agentic AI patterns. Each sub-project is self-contained and demonstrates a different technique or use case.

## Tech Stack

- **Language**: Python 3.11+
- **LLM Orchestration**: LangChain, LangGraph
- **Agent Frameworks**: SmolAgents
- **LLM Providers**: OpenAI (gpt-4o-mini), compatible with any OpenAI-API-compatible endpoint
- **Vector Store**: ChromaDB (`langchain-chroma`)
- **Embeddings**: OpenAI text-embedding-ada-002
- **Validation**: Pydantic v2

## Projects

### [`RAG-research-papers-assistant`](RAG-research-papers-assistant/)

A RAG pipeline that ingests a corpus of agentic AI research papers from a zip archive, stores them in a Chroma vector store, and answers questions grounded in the papers.

Includes an LLM-as-a-judge evaluation module that scores responses on **groundedness** (no hallucination) and **relevance** (answers the question).

| File | Purpose |
|---|---|
| `main.py` | Ingest pipeline + query interface |
| `evaluate.py` | LLM-as-a-judge evaluation (groundedness & relevance, 1–5) |

**Quickstart**

```bash
source .venv/bin/activate
cd RAG-research-papers-assistant

python main.py ingest                                    # run once
python main.py "What are the key challenges in agentic AI?"
python evaluate.py                                       # evaluate RAG quality
```

---

### [`claim-processing-agent`](claim-processing-agent/)

An autonomous insurance claim processing agent. A `ToolCallingAgent` from SmolAgents orchestrates a 5-step pipeline — validating coverage, retrieving relevant policy sections via RAG, and producing a structured claim decision.

Tools share state via a module-level dict so the LLM never needs to pass structured data between tool calls.

| File | Purpose |
|---|---|
| `models.py` | Pydantic models: `ClaimsInfo`, `PolicyRecommendation`, `ClaimDecision` |
| `rag.py` | Policy PDF ingestion + ChromaDB retrieval |
| `tools.py` | 5 SmolAgent tools + shared `_state` pipeline |
| `prompts.py` | System and planning prompts |
| `main.py` | CLI entry point |
| `tests/` | 50 pytest tests (tools + main) |

**Workflow:** `importClaims → validateClaims → getRelevantPolicyInfo → processClaim → finalDecision`

**Quickstart**

```bash
source .venv/bin/activate
cd claim-processing-agent

python main.py claims/sample_claim.json    # ingests policy.pdf on first run
```

---

## Repository Structure

```
RAG-Agentic-AI/
├── .venv/                              # Shared virtual environment
├── RAG-research-papers-assistant/
│   ├── main.py
│   ├── evaluate.py
│   ├── agentic_ai_research_papers.zip
│   └── .env
├── claim-processing-agent/
│   ├── policy_docs/policy.pdf
│   ├── dataset/coverage_data.csv
│   ├── claims/sample_claim.json
│   ├── models.py
│   ├── rag.py
│   ├── tools.py
│   ├── prompts.py
│   ├── main.py
│   └── tests/
└── .gitignore
```

## Setup

```bash
git clone <repo-url>
cd RAG-Agentic-AI
python3 -m venv .venv
source .venv/bin/activate
pip install langchain langchain-community langchain-chroma langchain-openai \
            langchain-text-splitters chromadb pypdf tiktoken pandas \
            python-dotenv smolagents tenacity pydantic pytest
```

Each project needs its own `.env` — keys must be **unquoted**:
```
OPENAI_API_KEY=your-key-here
OPENAI_API_BASE=https://api.openai.com/v1
```
