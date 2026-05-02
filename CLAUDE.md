# CLAUDE.md

## Repository overview

This is a multi-project Python monorepo for RAG and Agentic AI experiments. Each project lives in its own subdirectory and is independently runnable. There is a single shared `.venv` at the repo root.

## Shared virtual environment

All projects use `/RAG-Agentic-AI/.venv`. Always use this interpreter:

```bash
source .venv/bin/activate
# or explicitly
/Users/sanafarooqui/Documents/GitHub/RAG-Agentic-AI/.venv/bin/python
```

## Installing dependencies

```bash
.venv/bin/pip install <package>
```

## Environment variables

Each project has its own `.env` file (not committed). The standard keys are:

```
OPENAI_API_KEY=...
OPENAI_API_BASE=...
```

`.env` files must use unquoted key names (`KEY=value`, not `"KEY"=value`).

`load_dotenv` is always called with an explicit `dotenv_path` relative to `__file__` so it works regardless of working directory:

```python
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
```

## Projects

### RAG-research-papers-assistant

- **`main.py`** — ingestion and query
  - `python main.py ingest` — extract zip, embed PDFs, persist Chroma DB
  - `python main.py "<question>"` — query the vector store
- **`evaluate.py`** — LLM-as-a-judge scoring (groundedness + relevance, 1–5)
  - `python evaluate.py` — runs default test questions
  - `python evaluate.py "question 1" "question 2"` — custom questions

Chroma telemetry is disabled via `os.environ["ANONYMIZED_TELEMETRY"] = "false"` in each entry point.

## Conventions

- LLM: `gpt-4o-mini` via `ChatOpenAI` from `langchain_openai`
- Embeddings: `text-embedding-ada-002` via `OpenAIEmbeddings`
- Vector store: `Chroma` from `langchain_chroma` (not `langchain_community`)
- Chains use LCEL (`|` operator) throughout
- Structured judge output uses `judge.with_structured_output(PydanticModel)`
- No comments on obvious code; comments only for non-obvious constraints

## Adding a new project

1. Create a new subdirectory: `mkdir <project-name>`
2. Add a `.env` (unquoted keys)
3. Import the shared venv — no new venv needed
4. Add an entry in `README.md` under **Projects**
