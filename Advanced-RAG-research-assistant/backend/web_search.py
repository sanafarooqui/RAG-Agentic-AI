import hashlib
import json

from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from backend.config import WEB_CACHE_TTL_SECONDS
from backend.redis_client import get_async_redis


async def web_search(query: str, max_results: int = 5) -> list[Document]:
    redis = get_async_redis()
    key = "websearch:" + hashlib.sha1(query.strip().lower().encode()).hexdigest()

    cached = await redis.get(key)
    if cached:
        results = json.loads(cached)
    else:
        response = await TavilySearch(max_results=max_results).ainvoke({"query": query})
        if not isinstance(response, dict) or "results" not in response:
            raise RuntimeError(f"Web search failed: {response.get('error') if isinstance(response, dict) else response}")
        results = response["results"]
        if results:
            await redis.set(key, json.dumps(results), ex=WEB_CACHE_TTL_SECONDS)

    return [
        Document(
            page_content=r.get("content", ""),
            metadata={"kind": "web", "title": r.get("title") or r.get("url", ""), "url": r.get("url", "")},
        )
        for r in results
        if r.get("content")
    ]
