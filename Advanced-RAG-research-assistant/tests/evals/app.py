import asyncio
import time

from deepeval.integrations.langchain import CallbackHandler
from deepeval.tracing import update_current_trace

import judge  # noqa: F401  (loads .env and eval environment before backend imports)
from backend import sessions
from backend.rag_graph import rag_graph

# The paper the golden set is generated from, as ingested by the app (document title = file stem)
PAPER_TITLE = "1706.03762v7"

# backend caches loop-bound async clients (Redis), so every run must share one event loop
_LOOP = asyncio.new_event_loop()


async def _doc_scope() -> tuple[list[str], list[str]]:
    records = [r for r in await sessions.get_documents() if r.title == PAPER_TITLE]
    return [r.doc_id for r in records], [r.title for r in records]


async def _run(question: str, history: list[dict], with_docs: bool) -> dict:
    doc_ids, titles = await _doc_scope() if with_docs else ([], [])
    state = {
        "question": question,
        "history": history,
        "doc_ids": doc_ids,
        "doc_titles": titles,
    }
    started = time.perf_counter()
    result = await rag_graph.ainvoke(state, config={"callbacks": [CallbackHandler()]})
    result["latency_s"] = time.perf_counter() - started
    update_current_trace(
        input=question,
        output=result["answer"],
        retrieval_context=[d.page_content for d in result.get("documents", [])],
    )
    return result


def run_rag(question: str, history: list[dict] | None = None, with_docs: bool = True) -> dict:
    """Run one question through the real graph, traced; the paper is the selected document."""
    return _LOOP.run_until_complete(_run(question, history or [], with_docs))
