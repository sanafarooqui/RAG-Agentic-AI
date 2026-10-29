from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field

Route = Literal["knowledge_base", "web_search", "general"]


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class RouteDecision(BaseModel):
    route: Route = Field(description="Where to answer the question from")
    standalone_question: str = Field(
        description="The question with conversation-dependent references resolved, otherwise unchanged"
    )


class RelevanceGrade(BaseModel):
    relevant: bool = Field(description="True if the passages contain information that answers the question")
    reason: str


class RerankResult(BaseModel):
    ranked_indices: list[int] = Field(description="Passage indices ordered from most to least useful")


class RewrittenQuery(BaseModel):
    query: str = Field(description="A better search query for the vector store")


class Source(BaseModel):
    id: int
    kind: Literal["local", "web"]
    title: str
    location: str = ""
    snippet: str


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    sources: list[Source] = []
    route: Optional[Route | Literal["fallback"]] = None
    created_at: str = Field(default_factory=utcnow)


class DocumentRecord(BaseModel):
    doc_id: str
    title: str
    origin: Literal["upload", "arxiv"]
    n_chunks: int
    arxiv_id: Optional[str] = None
    created_at: str = Field(default_factory=utcnow)


class Session(BaseModel):
    id: str
    title: str = "New chat"
    doc_ids: list[str] = []
    created_at: str = Field(default_factory=utcnow)
    updated_at: str = Field(default_factory=utcnow)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    force_route: Optional[Route] = None


class ArxivSearchRequest(BaseModel):
    query: str = Field(min_length=1)


class ArxivImportRequest(BaseModel):
    arxiv_id: str
    session_id: str


class SessionDocumentsRequest(BaseModel):
    doc_ids: list[str]
