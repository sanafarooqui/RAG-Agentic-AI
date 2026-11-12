import asyncio
import re
from typing import Literal, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from backend import vector_store
from backend.pii import mask_history, mask_text
from backend.config import (
    MAX_RETRIEVAL_ATTEMPTS,
    EXPAND_TOP_N,
    PRUNE_WINDOW,
    RERANK_TOP_N,
    TAVILY_ENABLED,
    get_llm,
)
from backend.models import (
    PrunedPassages,
    RelevanceGrade,
    RerankResult,
    RewrittenQuery,
    RouteDecision,
    Source,
)
from backend.web_search import web_search

FALLBACK_MESSAGE = (
    "I couldn't find enough reliable evidence to answer that. The selected documents don't "
    "cover it well, and no external sources supported an answer. Try rephrasing the question, "
    "adding more documents, or running a web search."
)


class RAGState(TypedDict, total=False):
    question: str
    history: list[dict]
    doc_ids: list[str]
    doc_titles: list[str]
    force_route: Optional[str]
    route: str
    standalone_question: str
    query: str
    attempts: int
    documents: list[Document]
    relevant: bool
    grade_reason: str
    web_fallback: bool
    answer: str
    sources: list[dict]


def _history_text(history: list[dict]) -> str:
    if not history:
        return "(no previous messages)"
    return "\n".join(f"{m['role']}: {m['content'][:800]}" for m in history)


def _format_passages(docs: list[Document]) -> str:
    return "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs))


def _build_sources(docs: list[Document]) -> list[dict]:
    sources = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata
        if meta.get("kind") == "web":
            kind, location = "web", meta.get("url", "")
        else:
            kind = "local"
            location = f"page {meta['page']}" if meta.get("page") else ""
        sources.append(
            Source(
                id=i,
                kind=kind,
                title=meta.get("title", "Untitled"),
                location=location,
                snippet=d.page_content[:400],
            ).model_dump(mode="json")
        )
    return sources


ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You route questions for a research-paper assistant.\n"
            "Routes:\n"
            "- knowledge_base: the question can be answered from the user's selected documents. When "
            "documents are selected, assume the user is asking about them, including questions that "
            "name the paper's subject without naming the file (the titles may be bare file names or "
            "arXiv ids). Choose this for any question about a model, method, setting or result that "
            "such a paper could contain.\n"
            "- web_search: the question clearly needs current or external information, or asks to "
            "verify or fact-check a claim, or no documents are selected and it needs sources.\n"
            "- general: casual conversation or general knowledge needing no sources.\n"
            "Selected documents: {titles}\n"
            "Also fill standalone_question. Only if the latest question depends on the conversation "
            "(pronouns such as 'it' or 'them', 'that paper', or an elliptical follow-up), resolve "
            "those references using the conversation. Do not add, remove or rephrase any other words. "
            "If the question already stands alone, return it exactly as written.",
        ),
        ("human", "Conversation so far:\n{history}\n\nLatest question: {question}"),
    ]
)

GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Decide whether the passages contain information that helps answer the question. "
            "Be strict: topical overlap without an actual answer is not relevant.",
        ),
        ("human", "Question: {question}\n\nPassages:\n{passages}"),
    ]
)

RERANK_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Order the passages by how useful they are for answering the question, best first. "
            "Return only indices of useful passages.",
        ),
        ("human", "Question: {question}\n\nPassages:\n{passages}"),
    ]
)

PRUNE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Each passage is split into numbered sentences. For each passage, return the numbers of "
            "the sentences that help answer the question: those that contain the answer, and the "
            "definitions, settings or reasons it depends on. Include a sentence if it might help. "
            "Leave a passage out only if none of it helps.",
        ),
        ("human", "Question: {question}\n\nPassages:\n{passages}"),
    ]
)

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "A vector-store search returned weak results. Write a better search query: use "
            "specific terminology likely to appear in a research paper, no filler words.",
        ),
        (
            "human",
            "Question: {question}\nPrevious query: {query}\nWhy results were weak: {reason}",
        ),
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a research assistant. Answer using ONLY the numbered context passages. "
            "Cite supporting passages inline like [1] or [2][3]. If the context only partly "
            "answers the question, say what is missing. Never invent facts. If the question assumes "
            "something the passages contradict, say the assumption is wrong and give what the "
            "passages say instead of answering as if it were true.\n{origin_note}",
        ),
        # No chat history here: the question is already standalone, and earlier answers are not evidence
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)

GENERAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful research assistant. Answer from your own general knowledge. "
            "Start with: \"(From general model knowledge, not from your documents or web sources.)\" "
            "and be upfront about uncertainty.",
        ),
        ("human", "Conversation so far:\n{history}\n\nQuestion: {question}"),
    ]
)

LOCAL_NOTE = "The passages come from the user's uploaded research documents."
WEB_NOTE = (
    "The passages come from EXTERNAL web sources, not the user's documents. State clearly "
    "that the information is from the web."
)
WEB_FALLBACK_NOTE = (
    "The user's documents did not cover this, so the passages come from EXTERNAL web sources. "
    "Start by saying the selected documents do not report this and that what follows is from the "
    "web. Attribute each claim to its web source, say when a figure is a third-party estimate, and "
    "never describe the user's paper as doing, reporting or using something unless a passage says "
    "so. If the web passages do not directly answer the question, say so instead of guessing."
)


async def mask_node(state: RAGState) -> RAGState:
    # First node, so PII never reaches the router, the vector-store query, the web search or the prompts
    return {"question": mask_text(state["question"]), "history": mask_history(state.get("history", []))}


async def route_node(state: RAGState) -> RAGState:
    question = state["question"]
    forced = state.get("force_route")
    if forced:
        return {"route": forced, "standalone_question": question, "attempts": 0}

    decision = await (ROUTER_PROMPT | get_llm().with_structured_output(RouteDecision)).ainvoke(
        {
            "titles": ", ".join(state.get("doc_titles", [])) or "(none selected)",
            "history": _history_text(state.get("history", [])),
            "question": question,
        }
    )
    route = decision.route
    if route == "knowledge_base" and not state.get("doc_ids"):
        route = "web_search" if TAVILY_ENABLED else "general"
    if route == "web_search" and state.get("doc_ids"):
        # With documents selected, look there first; weak retrieval still falls back to the web.
        # Only an explicit web search (force_route) skips the documents.
        route = "knowledge_base"
    if route == "web_search" and not TAVILY_ENABLED:
        route = "general"
    # With no history there is nothing to resolve; rewording only risks shifting the embedding
    standalone = decision.standalone_question if state.get("history") else question
    return {"route": route, "standalone_question": standalone, "attempts": 0}


async def retrieve_node(state: RAGState) -> RAGState:
    query = state.get("query") or state["standalone_question"]
    docs = await vector_store.search(query, state["doc_ids"])
    return {"query": query, "documents": docs, "attempts": state.get("attempts", 0) + 1}


async def grade_node(state: RAGState) -> RAGState:
    docs = state["documents"]
    if not docs:
        return {"relevant": False, "grade_reason": "No passages were retrieved"}
    grade = await (GRADE_PROMPT | get_llm().with_structured_output(RelevanceGrade)).ainvoke(
        {"question": state["standalone_question"], "passages": _format_passages(docs)}
    )
    return {"relevant": grade.relevant, "grade_reason": grade.reason}


async def rewrite_node(state: RAGState) -> RAGState:
    rewritten = await (REWRITE_PROMPT | get_llm().with_structured_output(RewrittenQuery)).ainvoke(
        {
            "question": state["standalone_question"],
            "query": state["query"],
            "reason": state.get("grade_reason", ""),
        }
    )
    return {"query": rewritten.query}


async def rerank_node(state: RAGState) -> RAGState:
    docs = state["documents"]
    if len(docs) <= RERANK_TOP_N:
        return {}
    result = await (RERANK_PROMPT | get_llm().with_structured_output(RerankResult)).ainvoke(
        {"question": state["standalone_question"], "passages": _format_passages(docs)}
    )
    seen: list[int] = []
    for i in result.ranked_indices:
        if 0 <= i < len(docs) and i not in seen:
            seen.append(i)
    # Pad with the original retrieval order if the model returned too few indices
    seen += [i for i in range(len(docs)) if i not in seen]
    return {"documents": [docs[i] for i in seen[:RERANK_TOP_N]]}


async def expand_node(state: RAGState) -> RAGState:
    docs = state["documents"]
    have = {(d.metadata.get("doc_id"), d.metadata.get("chunk_index")) for d in docs}
    wanted: dict[str, set[int]] = {}
    for d in docs[:EXPAND_TOP_N]:
        doc_id, idx = d.metadata.get("doc_id"), d.metadata.get("chunk_index")
        if doc_id is None or idx is None:
            continue
        for n in (idx - 1, idx + 1):
            if (doc_id, n) not in have:
                wanted.setdefault(doc_id, set()).add(n)
    neighbours: list[Document] = []
    for doc_id, indexes in wanted.items():
        neighbours += await asyncio.to_thread(vector_store.get_chunks, doc_id, sorted(indexes))
    # Neighbours go after the reranked passages; pruning drops whatever does not help
    for n in neighbours:
        n.metadata = {**n.metadata, "neighbour": True}
    return {"documents": docs + neighbours}


def _table_like(text: str) -> bool:
    # PyMuPDF flattens tables into columns of short numeric cells, which an LLM pruner treats as noise
    tokens = text.split()
    numeric = sum(1 for t in tokens if re.fullmatch(r"[\d.,×%()+-]+", t))
    return len(tokens) >= 20 and numeric / len(tokens) > 0.3


def _squash(text: str) -> str:
    return " ".join(text.split())


def _sentences(text: str) -> list[str]:
    return re.split(r"(?<=[.!?])\s+(?=[A-Z(\[])", _squash(text))


def _numbered(docs: list[Document], split: dict[int, list[str]]) -> str:
    blocks = []
    for i, d in enumerate(docs):
        lines = "\n".join(f"  {j}. {s}" for j, s in enumerate(split[i]))
        blocks.append(f"Passage {i}:\n{lines}")
    return "\n\n".join(blocks)


def _keep_with_window(sentences: list[str], chosen: list[int]) -> str:
    keep = sorted({k for c in chosen for k in range(c - PRUNE_WINDOW, c + PRUNE_WINDOW + 1) if 0 <= k < len(sentences)})
    parts, run = [], [keep[0]]
    for k in keep[1:]:
        if k == run[-1] + 1:
            run.append(k)
        else:
            parts.append(" ".join(sentences[i] for i in run))
            run = [k]
    parts.append(" ".join(sentences[i] for i in run))
    return "\n[...]\n".join(parts)


async def prune_node(state: RAGState) -> RAGState:
    docs = state["documents"]
    # PyMuPDF flattens tables into columns of numeric cells; keep reranked table passages whole, and
    # table neighbours that continue one, since sentence selection cannot read them
    tables = {i for i, d in enumerate(docs) if _table_like(d.page_content)}
    whole = tables if any(not docs[i].metadata.get("neighbour") for i in tables) else set()
    split = {i: _sentences(d.page_content) for i, d in enumerate(docs) if i not in whole}
    pruned: dict[int, Document] = {i: docs[i] for i in whole}
    if split:
        result = await (PRUNE_PROMPT | get_llm().with_structured_output(PrunedPassages)).ainvoke(
            {
                "question": state["standalone_question"],
                "passages": _numbered([docs[i] for i in split], dict(enumerate(split.values()))),
            }
        )
        order = list(split)
        for item in result.passages:
            if not 0 <= item.index < len(order):
                continue
            i = order[item.index]
            chosen = [c for c in item.sentences if 0 <= c < len(split[i])]
            if chosen and i not in pruned:
                pruned[i] = Document(page_content=_keep_with_window(split[i], chosen), metadata=docs[i].metadata)
    # Keep the reranked order, and never answer from nothing if pruning misbehaved
    ordered = [pruned[i] for i in sorted(pruned)]
    return {"documents": ordered or docs}


async def web_search_node(state: RAGState) -> RAGState:
    try:
        documents = await web_search(state["standalone_question"])
    except Exception:
        # A failed search must end in the "no evidence" answer, not abort the request
        documents = []
    return {
        "documents": documents,
        "web_fallback": state["route"] == "knowledge_base",
        "route": "web_search",
    }


async def generate_node(state: RAGState, config: RunnableConfig) -> RAGState:
    docs = state["documents"]
    if state.get("web_fallback"):
        note = WEB_FALLBACK_NOTE
    elif state["route"] == "web_search":
        note = WEB_NOTE
    else:
        note = LOCAL_NOTE
    context = "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs, start=1))
    msg = await (ANSWER_PROMPT | get_llm()).ainvoke(
        {
            "origin_note": note,
            "context": context,
            "question": state["standalone_question"],
        },
        config,
    )
    return {"answer": msg.content, "sources": _build_sources(docs)}


async def general_node(state: RAGState, config: RunnableConfig) -> RAGState:
    msg = await (GENERAL_PROMPT | get_llm()).ainvoke(
        {"history": _history_text(state.get("history", [])), "question": state["question"]},
        config,
    )
    return {"answer": msg.content, "sources": []}


async def fallback_node(state: RAGState) -> RAGState:
    return {"route": "fallback", "answer": FALLBACK_MESSAGE, "sources": []}


def after_route(state: RAGState) -> Literal["retrieve", "web_search", "general"]:
    return {"knowledge_base": "retrieve", "web_search": "web_search", "general": "general"}[state["route"]]


def after_grade(state: RAGState) -> Literal["rerank", "rewrite", "web_search", "fallback"]:
    if state["relevant"]:
        return "rerank"
    if state["attempts"] < MAX_RETRIEVAL_ATTEMPTS:
        return "rewrite"
    return "web_search" if TAVILY_ENABLED else "fallback"


def after_web_search(state: RAGState) -> Literal["generate", "fallback"]:
    return "generate" if state.get("documents") else "fallback"


def build_graph():
    g = StateGraph(RAGState)
    g.add_node("mask", mask_node)
    g.add_node("route", route_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("grade", grade_node)
    g.add_node("rewrite", rewrite_node)
    g.add_node("rerank", rerank_node)
    g.add_node("expand", expand_node)
    g.add_node("prune", prune_node)
    g.add_node("web_search", web_search_node)
    g.add_node("generate", generate_node)
    g.add_node("general", general_node)
    g.add_node("fallback", fallback_node)

    g.add_edge(START, "mask")
    g.add_edge("mask", "route")
    g.add_conditional_edges("route", after_route)
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", after_grade)
    g.add_edge("rewrite", "retrieve")
    g.add_edge("rerank", "expand")
    g.add_edge("expand", "prune")
    g.add_edge("prune", "generate")
    g.add_conditional_edges("web_search", after_web_search)
    g.add_edge("generate", END)
    g.add_edge("general", END)
    g.add_edge("fallback", END)
    return g.compile()


rag_graph = build_graph()
