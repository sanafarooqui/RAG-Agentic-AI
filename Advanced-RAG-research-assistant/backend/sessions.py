import uuid
from datetime import datetime

from backend.config import HISTORY_WINDOW
from backend.models import DocumentRecord, Message, Session, utcnow
from backend.redis_client import get_async_redis

SESSIONS_INDEX = "sessions"
DOCS_INDEX = "documents"


def _session_key(sid: str) -> str:
    return f"session:{sid}"


def _messages_key(sid: str) -> str:
    return f"session:{sid}:messages"


async def create_session() -> Session:
    session = Session(id=uuid.uuid4().hex)
    await save_session(session)
    return session


async def save_session(session: Session) -> None:
    redis = get_async_redis()
    session.updated_at = utcnow()
    await redis.set(_session_key(session.id), session.model_dump_json())
    await redis.zadd(SESSIONS_INDEX, {session.id: _timestamp(session.updated_at)})


async def get_session(sid: str) -> Session | None:
    raw = await get_async_redis().get(_session_key(sid))
    return Session.model_validate_json(raw) if raw else None


async def list_sessions() -> list[Session]:
    redis = get_async_redis()
    ids = await redis.zrevrange(SESSIONS_INDEX, 0, -1)
    if not ids:
        return []
    raws = await redis.mget([_session_key(i) for i in ids])
    return [Session.model_validate_json(r) for r in raws if r]


async def delete_session(sid: str) -> None:
    redis = get_async_redis()
    await redis.delete(_session_key(sid), _messages_key(sid))
    await redis.zrem(SESSIONS_INDEX, sid)


async def add_messages(sid: str, *messages: Message) -> None:
    # One RPUSH so a question and its answer are stored together or not at all
    await get_async_redis().rpush(_messages_key(sid), *(m.model_dump_json() for m in messages))


async def get_messages(sid: str, last: int | None = None) -> list[Message]:
    start = -last if last else 0
    raws = await get_async_redis().lrange(_messages_key(sid), start, -1)
    return [Message.model_validate_json(r) for r in raws]


async def get_history(sid: str) -> list[dict]:
    messages = await get_messages(sid, last=HISTORY_WINDOW)
    return [{"role": m.role, "content": m.content} for m in messages]


async def save_document(record: DocumentRecord) -> None:
    await get_async_redis().hset(DOCS_INDEX, record.doc_id, record.model_dump_json())


async def get_documents(doc_ids: list[str] | None = None) -> list[DocumentRecord]:
    redis = get_async_redis()
    if doc_ids is None:
        raws = list((await redis.hgetall(DOCS_INDEX)).values())
    else:
        raws = [r for r in await redis.hmget(DOCS_INDEX, doc_ids) if r] if doc_ids else []
    records = [DocumentRecord.model_validate_json(r) for r in raws]
    return sorted(records, key=lambda r: r.created_at, reverse=True)


def _timestamp(iso: str) -> float:
    return datetime.fromisoformat(iso).timestamp()
