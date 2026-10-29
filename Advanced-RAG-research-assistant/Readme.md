# Advanced RAG Research Assistant

Agentic RAG chatbot for research papers. Phase 1: local demo (FastAPI + LangGraph backend, React frontend, Redis). See [Intructions.md](Intructions.md) for the full spec.

## Layout

```
backend/
  api.py           FastAPI routes (sessions, upload, ArXiv, SSE chat)
  rag_graph.py     LangGraph: route → retrieve → grade → rewrite/rerank → generate
  paper_loader.py  File + ArXiv ingestion and chunking
  vector_store.py  Qdrant + Redis-cached embeddings
  sessions.py      Sessions, messages and document library in Redis
  web_search.py    Tavily search with Redis result cache
  models.py        Pydantic models
  config.py        Env, paths, tunables
frontend/          Vite + React + TypeScript UI
main.py            Backend entry point
```

## Run

1. Start Redis: `redis-server --daemonize yes` or `docker compose up -d redis`
2. `.env` in this folder: `OPENAI_API_KEY`, `OPENAI_API_BASE_URL`, `OPENAI_MODEL`, `TAVILY_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY` (optional `QDRANT_COLLECTION`, default `research_papers`; optional `REDIS_URL`, default `redis://localhost:6379/0`)
3. Backend (from the project root venv): `../.venv/bin/python main.py` → http://127.0.0.1:8000
4. Frontend: `cd frontend && npm install && npm run dev` → http://localhost:5173

## How a question is handled

1. **Router** picks `knowledge_base`, `web_search` or `general`, and rewrites the question using recent history (short-term memory).
2. **Knowledge base**: retrieve top 8 chunks from the session's documents, grade relevance, rewrite the query and retry once if weak, then rerank to the top 4.
3. **Fallback**: still weak → Tavily web search → if that also finds nothing, an explicit "insufficient evidence" answer.
4. **Answer** cites passages as `[n]`; each source is tagged as a local document (with page) or a web URL, and the UI labels where the answer came from.

The web search box in the sidebar forces the `web_search` route, which also works for verifying claims.
