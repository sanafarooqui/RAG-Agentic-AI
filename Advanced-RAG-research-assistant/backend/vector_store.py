import uuid
from functools import lru_cache

from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_community.storage import RedisStore
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from backend.config import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    QDRANT_URL,
    RETRIEVE_K,
    get_base_embeddings,
)
from backend.redis_client import get_sync_redis

DOC_ID_FIELD = "metadata.doc_id"


@lru_cache
def get_embeddings() -> CacheBackedEmbeddings:
    store = RedisStore(client=get_sync_redis(), namespace="embcache")
    return CacheBackedEmbeddings.from_bytes_store(
        get_base_embeddings(),
        store,
        namespace=EMBEDDING_MODEL,
        query_embedding_cache=store,
        key_encoder="sha256",
    )


@lru_cache
def get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


@lru_cache
def get_vector_store() -> QdrantVectorStore:
    client = get_client()
    if not client.collection_exists(QDRANT_COLLECTION):
        client.create_collection(
            QDRANT_COLLECTION,
            vectors_config=models.VectorParams(size=EMBEDDING_DIM, distance=models.Distance.COSINE),
        )
    # Qdrant Cloud rejects filtering on a field without a payload index
    client.create_payload_index(
        QDRANT_COLLECTION, field_name=DOC_ID_FIELD, field_schema=models.PayloadSchemaType.KEYWORD
    )
    return QdrantVectorStore(client=client, collection_name=QDRANT_COLLECTION, embedding=get_embeddings())


def _doc_filter(doc_ids: list[str]) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key=DOC_ID_FIELD, match=models.MatchAny(any=doc_ids))])


def _point_id(doc_id: str, index: int) -> str:
    # Qdrant point ids must be UUIDs or unsigned ints; derive stable ones so re-ingesting overwrites
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}-{index}"))


def has_document(doc_id: str) -> bool:
    store = get_vector_store()
    return store.client.count(QDRANT_COLLECTION, count_filter=_doc_filter([doc_id]), exact=True).count > 0


def add_chunks(doc_id: str, chunks: list[Document]) -> None:
    ids = [_point_id(doc_id, i) for i in range(len(chunks))]
    get_vector_store().add_documents(chunks, ids=ids)


def delete_document(doc_id: str) -> None:
    get_client().delete(QDRANT_COLLECTION, points_selector=models.FilterSelector(filter=_doc_filter([doc_id])))


async def search(query: str, doc_ids: list[str], k: int = RETRIEVE_K) -> list[Document]:
    if not doc_ids:
        return []
    return await get_vector_store().asimilarity_search(query, k=k, filter=_doc_filter(doc_ids))
