# CLAUDE.md

Project-specific notes for `Advanced-RAG-research-assistant`. Repo-wide rules (shared `.venv`, unquoted `.env` keys, LCEL, Pydantic v2) are in the root `CLAUDE.md`. Spec is in `Intructions.md`; scope is Phase 1 (local demo). Evaluation, Docker image and AWS deployment are Phase 2.

## Run

- Redis must be running: `redis-server --daemonize yes` or `docker compose up -d redis`
- Backend: `../.venv/bin/python main.py` (port 8000). The backend pings Redis on startup and exits if it is down.
- Frontend: `cd frontend && npm install && npm run dev` (port 5173, proxies `/api` to 8000)
- `.env` changes need a backend restart: `load_dotenv` runs once at import, and uvicorn's reload only watches `.py` files.

## Deviations from repo-wide conventions

- Vector store is **Qdrant** (`langchain_qdrant`), not Chroma. Env: `QDRANT_URL`, `QDRANT_API_KEY`, optional `QDRANT_COLLECTION` (default `research_papers`).
- `.env` uses `OPENAI_API_BASE_URL` and `OPENAI_MODEL`, not `OPENAI_API_BASE`. Also `TAVILY_API_KEY`, optional `REDIS_URL`.

## Architecture

- `backend/rag_graph.py` (LangGraph): route → retrieve → grade → (rewrite → retrieve, up to `MAX_RETRIEVAL_ATTEMPTS`) → rerank → generate. Weak retrieval falls back to Tavily, then to a "no evidence" answer.
- `RAGState` keeps three versions of the question on purpose: `question` (user's exact words), `standalone_question` (history-dependent references resolved; used for retrieval, grading and the answer), `query` (the vector-store search string; only `rewrite_node` changes it after a weak retrieval).
- `route_node` uses the raw question as `standalone_question` when there is no history, and the router prompt tells the model to resolve references only and not reword anything else. (Added after a rewrite shifted the embedding enough to drop a relevant chunk; not yet re-tested end to end.)
- Redis holds sessions, messages (`session:{id}:messages`), the document library (`documents` hash), the embedding cache (`embcache*`, SHA-256 keys) and Tavily results (1h TTL, never cached when empty or failed).
- A chat turn is persisted only after the graph succeeds: the user message and the answer are written together (`sessions.add_messages`, one RPUSH), and the session title is set then too. A failed or abandoned request stores nothing, so there are no orphaned user messages or duplicates on retry.
- One shared Qdrant collection; each session filters by its `doc_ids` on the indexed payload field `metadata.doc_id`. Filtering on an unindexed field returns a 400 on Qdrant Cloud.
- Point IDs are `uuid5(doc_id-chunk_index)`, so re-ingesting overwrites instead of duplicating.

## Chunking (`paper_loader.build_chunks`)

- PDFs load with `PyMuPDFLoader`. Pages are joined and split as one text, not page by page. Splitting per page forced a cut at every page break; each chunk's page is derived from its start offset.
- A section-heading regex is the first splitter separator. PyMuPDF puts a heading's number and title on separate lines, and table rows like `91.7 Transformer (4 layers)` look like headings unless titles are limited to letters.
- Each chunk gets metadata `chunk_index` and `section`, and its text is prefixed `[title | section]` so the heading is part of what is embedded.
- Document IDs hash the text only and `has_document` skips existing IDs. After changing chunking, delete the document's points and re-import, or old chunks are reused.

## Learning: "How many layers are in the encoder and decoder stacks?" (Attention Is All You Need)

The answer gave the encoder (6) but said the decoder count was missing. Traced cause, in order:

1. Chunking split section 3.1 so the encoder paragraph and the decoder paragraph were in adjacent chunks, and the decoder chunk began mid-sentence ("itself. To facilitate these residual connections...") with no hint of the heading. Its embedding looked like residuals and masking, not layer counts.
2. The router rewrote the question to "...of the Transformer model?". For the raw question the decoder chunk ranked 3rd of 8; for the rewritten one it fell out of the top 8.
3. The grader marked the passages relevant because the encoder chunk was present, even reasoning that the decoder "implies a similar structure", so no rewrite or retry happened.
4. Rerank only reorders what was retrieved, so it could not recover the missing chunk.

Done so far: whole-document splitting, heading-aware separator and `[title | section]` prefix (section 3.1 now lands in one chunk), and no-rewrite-without-history.

Not done yet, in suggested order: moderate chunk size increase (about 1,200-1,500 chars, overlap 250), neighbour expansion after rerank using `chunk_index`, grader that checks the question is fully answered and names what is missing (feeds the rewrite), retrieve about 20 then rerank to 5-6, sub-query decomposition for multi-part questions, hybrid (sparse + dense) search. This was one question on one paper; confirm changes on other questions before tuning numbers, ideally with the Phase 2 golden set.

## Learning: chat history leaking into grounded answers

Asking "How does the Transformer compare to GPT-4?" right after a question about learned vs sinusoidal positional embeddings returned an answer about positional embeddings, labelled "From the web" and citing the web sources for a claim that actually came from the previous assistant message. The web search itself was fine (all five sources were GPT-4 pages). Cause: `ANSWER_PROMPT` included "Conversation so far", so the model treated earlier turns as evidence; an earlier router prompt that rewrote the question could also have pulled in the previous topic. Reproduced with the same web results: with history the answer hedged ("does not directly compare"), without it the answer was a proper grounded comparison.

Rule: the grounded-answer prompt gets only the numbered context and the standalone question, never the history. History is for the router (to resolve references) and the `general` route (no citations). `standalone_question` is not persisted, so check it in LangSmith when an answer drifts off topic.

## Testing and LangSmith

`.env` has `LANGSMITH_TRACING=true`, so any test run (scripts, `uvicorn` smoke tests, graph repros) sends traces to the user's real LangSmith project alongside their own chats.

- Prefix test commands with `LANGSMITH_TRACING=false`. `load_dotenv` does not override variables already in the environment, so this wins over `.env`.
- Stop a test server gracefully (SIGINT, then wait a few seconds) instead of killing it right after the response. The tracer posts a run's end in the background; an early kill leaves the run "pending" in LangSmith forever.
- Use the user's Docker Redis and Qdrant only for tests that need them, and delete any sessions or points the test created.
- If test traces do get through, tell the user which ones (time, input text) and ask before deleting any.

## Environment gotchas

- Python 3.14 on macOS: `backend/config.py` sets `SSL_CERT_FILE` from certifi, otherwise aiohttp (used by Tavily) fails certificate verification.
- `arxiv` 4.x has no `Result.download_pdf`; PDFs are downloaded from `result.pdf_url` with httpx.
- `docker compose up` fails with "Cannot connect to the Docker daemon" unless Docker Desktop is running (`open -a Docker`); Homebrew Redis (`brew services start redis`) works without it.
- Inspect Redis with `redis-cli --scan --pattern '*'`, or `redis-cli monitor` to watch commands live. Pipeline-level tracing is in LangSmith (`LANGSMITH_*` in `.env`).
