# CLAUDE.md

## Project overview

`claim-processing-agent` is an autonomous insurance claim processing pipeline. A `ToolCallingAgent` from SmolAgents orchestrates 5 tools in a fixed sequence. Tools share state via a module-level `_state` dict in `tools.py` — the LLM never passes data between tool calls.

## Virtual environment

Use the **shared root venv** — not a project-level one:
```bash
source /Users/sanafarooqui/Documents/GitHub/RAG-Agentic-AI/.venv/bin/activate
```

Always run pytest and Python with the venv explicitly if conda is also active:
```bash
/Users/sanafarooqui/Documents/GitHub/RAG-Agentic-AI/.venv/bin/pytest tests/ -v
/Users/sanafarooqui/Documents/GitHub/RAG-Agentic-AI/.venv/bin/python main.py claims/sample_claim.json
```

## Environment variables

Credentials live in `claim-processing-agent/.env`. Keys must be **unquoted**:
```
OPENAI_API_KEY=...       # not "OPENAI_API_KEY"=...
OPENAI_API_BASE=...
```

`load_dotenv` is called with an explicit `dotenv_path` relative to `__file__` in every entry point (`main.py`) so it resolves correctly from any working directory.

## Key design decisions

**Shared pipeline state (`_state` dict)**
Tools write their output to `tools._state` instead of returning values the LLM must pass forward. This is intentional — `ToolCallingAgent` uses OpenAI function calling and cannot reliably pass structured dicts between tool calls. Each tool reads what it needs from `_state`.

Call `reset_state()` before each agent run (already done in `main.py`) to prevent stale data from a previous run polluting the next.

**`ToolCallingAgent`, not `CodeAgent`**
The agent uses `ToolCallingAgent`. `system_prompt` is not a constructor argument — set it after construction:
```python
agent = ToolCallingAgent(tools=[...], model=model)
agent.prompt_templates["system_prompt"] = SYSTEM_PROMPT
```
`ToolCallingAgent` also automatically adds a `final_answer` built-in tool; tests use `.issubset()` to check registered tools.

**`max_steps=10`**
Passed to `agent.run()` to cap the agent at 10 steps and prevent infinite retry loops if a tool returns an error.

**ChromaDB telemetry**
`os.environ["ANONYMIZED_TELEMETRY"] = "false"` is set in `main.py` after `load_dotenv()` to suppress noisy telemetry warnings.

**Chroma import**
Use `from langchain_chroma import Chroma`, not `from langchain_community.vectorstores import Chroma` — the community version is deprecated.

## Running commands

```bash
# Process a claim (ingests on first run)
python main.py claims/sample_claim.json

# Re-ingest after updating policy.pdf
rm -rf chroma_db/ && python main.py claims/sample_claim.json

# Run ingest only
python -c "from rag import ingest; ingest()"

# Run tests
/path/to/.venv/bin/pytest tests/ -v
```

## File responsibilities

| File | Responsibility |
|---|---|
| `models.py` | Pydantic models: `ClaimsInfo`, `PolicyRecommendation`, `ClaimDecision` |
| `rag.py` | `ingest()` — loads PDF, chunks, persists to Chroma; `get_retriever()` — loads Chroma for querying |
| `tools.py` | 5 `@tool`-decorated functions + `_state` dict + `reset_state()` |
| `prompts.py` | `SYSTEM_PROMPT` (agent behaviour) + `PLANNING_PROMPT` (task template) |
| `main.py` | CLI entry point — ingest gate, `reset_state`, `build_agent`, `agent.run` |

## Testing conventions

- `OPENAI_API_KEY` is set to `"test-key"` at the top of each test file before any imports that trigger LangChain/OpenAI initialisation.
- `autouse` fixture `clean_state` calls `reset_state()` before and after every test.
- LLM calls (`_call_process_llm`, `_call_decision_llm`, `_generate_queries`) and `get_retriever` are always mocked — tests never hit real APIs.
- `tools.DATASET_PATH` is patched per-test using a `tmp_path` CSV fixture.
- `conftest.py` inserts the project root onto `sys.path`.
