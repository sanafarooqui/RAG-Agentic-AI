import asyncio
import json
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from backend import paper_loader, sessions
from backend.config import ALLOWED_EXTENSIONS, TAVILY_ENABLED, UPLOAD_DIR
from backend.models import (
    ArxivImportRequest,
    ArxivSearchRequest,
    ChatRequest,
    DocumentRecord,
    Message,
    Session,
    SessionDocumentsRequest,
    Source,
)
from backend.rag_graph import rag_graph
from backend.redis_client import get_async_redis

STREAMING_NODES = {"generate", "general"}
NODE_STATUS = {
    "route": "Routing question",
    "retrieve": "Searching documents",
    "grade": "Checking relevance",
    "rewrite": "Rewriting query",
    "rerank": "Reranking passages",
    "web_search": "Searching the web",
    "generate": "Writing answer",
    "general": "Writing answer",
}


@asynccontextmanager
async def lifespan(_: FastAPI):
    await get_async_redis().ping()
    yield


app = FastAPI(title="Advanced RAG Research Assistant", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _require_session(sid: str) -> Session:
    session = await sessions.get_session(sid)
    if session is None:
        raise HTTPException(404, "Session not found")
    return session


async def _attach_document(session: Session, record: DocumentRecord) -> DocumentRecord:
    await sessions.save_document(record)
    if record.doc_id not in session.doc_ids:
        session.doc_ids.append(record.doc_id)
        await sessions.save_session(session)
    return record


@app.get("/api/health")
async def health():
    return {"status": "ok", "web_search": TAVILY_ENABLED}


@app.get("/api/sessions")
async def list_sessions():
    return await sessions.list_sessions()


@app.post("/api/sessions")
async def create_session():
    return await sessions.create_session()


@app.get("/api/sessions/{sid}")
async def get_session(sid: str):
    session = await _require_session(sid)
    return {
        "session": session,
        "messages": await sessions.get_messages(sid),
        "documents": await sessions.get_documents(session.doc_ids),
    }


@app.delete("/api/sessions/{sid}")
async def delete_session(sid: str):
    await _require_session(sid)
    await sessions.delete_session(sid)
    return {"deleted": sid}


@app.put("/api/sessions/{sid}/documents")
async def set_session_documents(sid: str, body: SessionDocumentsRequest):
    session = await _require_session(sid)
    known = {d.doc_id for d in await sessions.get_documents(body.doc_ids)}
    session.doc_ids = [d for d in body.doc_ids if d in known]
    await sessions.save_session(session)
    return session


@app.get("/api/documents")
async def list_documents():
    return await sessions.get_documents()


@app.post("/api/documents/upload")
async def upload_document(session_id: str = Form(...), file: UploadFile = File(...)):
    session = await _require_session(session_id)
    filename = Path(file.filename or "").name
    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, "Unsupported file type. Upload a PDF, Markdown or text file.")

    dest = UPLOAD_DIR / filename
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    try:
        record = await asyncio.to_thread(paper_loader.ingest, dest, Path(filename).stem, "upload")
    except ValueError as e:
        raise HTTPException(400, str(e))
    return await _attach_document(session, record)


@app.post("/api/arxiv/search")
async def arxiv_search(body: ArxivSearchRequest):
    try:
        return await asyncio.to_thread(paper_loader.arxiv_search, body.query)
    except Exception as e:
        raise HTTPException(502, f"ArXiv search failed: {e}")


@app.post("/api/arxiv/import")
async def arxiv_import(body: ArxivImportRequest):
    session = await _require_session(body.session_id)
    try:
        record = await asyncio.to_thread(paper_loader.import_arxiv, body.arxiv_id)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(502, f"ArXiv import failed: {e}")
    return await _attach_document(session, record)


def _event(name: str, payload: dict) -> dict:
    return {"event": name, "data": json.dumps(payload)}


@app.post("/api/sessions/{sid}/chat")
async def chat(sid: str, body: ChatRequest):
    session = await _require_session(sid)
    if body.force_route == "web_search" and not TAVILY_ENABLED:
        raise HTTPException(400, "Web search is not configured (missing TAVILY_API_KEY)")

    history = await sessions.get_history(sid)
    records = await sessions.get_documents(session.doc_ids)

    state = {
        "question": body.message,
        "history": history,
        "doc_ids": session.doc_ids,
        "doc_titles": [r.title for r in records],
        "force_route": body.force_route,
    }

    async def stream():
        final: dict = {}
        try:
            async for mode, chunk in rag_graph.astream(state, stream_mode=["updates", "messages"]):
                if mode == "messages":
                    msg, meta = chunk
                    if meta.get("langgraph_node") in STREAMING_NODES and msg.content:
                        yield _event("token", {"text": msg.content})
                else:
                    for node, update in chunk.items():
                        final.update(update or {})
                        if node in NODE_STATUS:
                            yield _event("status", {"node": node, "label": NODE_STATUS[node]})
        except Exception as e:
            yield _event("error", {"detail": str(e)})
            return

        message = Message(
            role="assistant",
            content=final.get("answer", ""),
            sources=[Source(**s) for s in final.get("sources", [])],
            route=final.get("route"),
        )
        # Persist only after a successful run, so a failed or abandoned request leaves no orphaned
        # user message. Re-read the session: documents may have been added while this was streaming.
        current = await sessions.get_session(sid)
        title = current.title if current else session.title
        if current:
            await sessions.add_messages(sid, Message(role="user", content=body.message), message)
            if current.title == "New chat":
                current.title = title = body.message[:60]
            await sessions.save_session(current)
        yield _event("done", {"message": message.model_dump(mode="json"), "title": title})

    return EventSourceResponse(stream())
