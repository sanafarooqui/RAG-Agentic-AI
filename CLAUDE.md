# CLAUDE.md

## Repository overview

Multi-project Python monorepo for RAG and Agentic AI experiments. Each project lives in its own subdirectory and is independently runnable. There is a single shared `.venv` at the repo root.

## Shared virtual environment

All projects use the root `.venv`. If conda is also active (`(base)` in prompt), the venv may not take precedence — use the full path to be safe:

```bash
source .venv/bin/activate
# or explicitly
/Users/sanafarooqui/Documents/GitHub/RAG-Agentic-AI/.venv/bin/python
/Users/sanafarooqui/Documents/GitHub/RAG-Agentic-AI/.venv/bin/pytest tests/ -v
```

Never create a project-level `.venv` — use the root one for all projects.

## Installing dependencies

```bash
.venv/bin/pip install <package>
```

## Environment variables

Each project has its own `.env` (not committed). Keys must be **unquoted**:
```
OPENAI_API_KEY=...        # correct
"OPENAI_API_KEY"=...      # WRONG — python-dotenv won't parse quoted key names
OPENAI_API_BASE=...
```

Always load `.env` with an explicit path so it resolves correctly from any working directory:
```python
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
```

## Conventions across all projects

- LLM: `gpt-4o-mini` via `ChatOpenAI` from `langchain_openai`
- Embeddings: `text-embedding-ada-002` via `OpenAIEmbeddings`
- Vector store: `from langchain_chroma import Chroma` — never `langchain_community` (deprecated)
- Chroma telemetry: `os.environ["ANONYMIZED_TELEMETRY"] = "false"` in every entry point, after `load_dotenv()`
- Chains: LCEL `|` operator throughout
- Validation: Pydantic v2 (`model_dump(mode="json")`, `model_validator(mode="after")`)
- No comments on obvious code; only for non-obvious constraints

## Projects

### RAG-research-papers-assistant

- `python main.py ingest` — extract zip, embed PDFs, persist Chroma DB
- `python main.py "<question>"` — query the vector store
- `python evaluate.py` — LLM-as-a-judge scoring (groundedness + relevance, 1–5)

LLM-as-a-judge uses `judge.with_structured_output(PydanticModel)` for typed scores.

---

### claim-processing-agent

- `python main.py claims/sample_claim.json` — process a claim end-to-end
- `python -c "from rag import ingest; ingest()"` — re-ingest after policy PDF changes
- `.venv/bin/pytest tests/ -v` — run 50-test suite

**SmolAgents `ToolCallingAgent` patterns learned:**

- `system_prompt` is not a constructor argument. Set it after construction:
  ```python
  agent = ToolCallingAgent(tools=[...], model=model)
  agent.prompt_templates["system_prompt"] = MY_PROMPT
  ```
- `ToolCallingAgent` automatically registers a built-in `final_answer` tool — tests checking registered tools should use `.issubset()`, not `==`.
- `ToolCallingAgent` uses OpenAI function calling. The LLM cannot reliably pass structured dicts between tool calls. Use **shared pipeline state** (`_state` dict in `tools.py`) — tools write output to state, subsequent tools read from it. Never rely on the LLM to chain tool outputs.
- Set `max_steps=N` on `agent.run()` to prevent infinite retry loops when tools fail.
- Call `reset_state()` before each `agent.run()` to clear stale state from prior runs.

**Chunking strategy for policy PDFs:**
`RecursiveCharacterTextSplitter` with `chunk_size=1000, chunk_overlap=200` and default separators. Policy documents have numbered clauses — double newlines naturally split sections.

**Testing conventions:**
- Set `os.environ["OPENAI_API_KEY"] = "test-key"` before any imports that trigger LangChain/OpenAI init.
- `autouse` fixture calls `reset_state()` before and after every test.
- Always mock LLM calls (`_call_process_llm`, `_call_decision_llm`, `_generate_queries`) and `get_retriever` — tests never hit real APIs.
- Patch `tools.DATASET_PATH` per-test using a `tmp_path` CSV fixture.

## Adding a new project

1. `mkdir <project-name>`
2. Add `<project-name>/.env` (unquoted keys)
3. Use the root `.venv` — no new venv needed
4. Update `README.md` under **Projects** and this file under **Projects**
