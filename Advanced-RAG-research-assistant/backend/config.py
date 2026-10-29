import os
from functools import lru_cache
from pathlib import Path

import certifi
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")
# python.org builds on macOS ship without a CA bundle, which breaks aiohttp-based clients (Tavily)
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

UPLOAD_DIR = BASE_DIR / "documents" / "uploads"
ARXIV_DIR = BASE_DIR / "documents" / "arxiv"
for _d in (UPLOAD_DIR, ARXIV_DIR):
    _d.mkdir(parents=True, exist_ok=True)

OPENAI_BASE_URL = os.getenv("OPENAI_API_BASE_URL") or os.getenv("OPENAI_API_BASE") or None
LLM_MODEL = os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_DIM = 1536
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "research_papers")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
TAVILY_ENABLED = bool(os.getenv("TAVILY_API_KEY"))

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVE_K = 8
RERANK_TOP_N = 4
MAX_RETRIEVAL_ATTEMPTS = 2
HISTORY_WINDOW = 6
WEB_CACHE_TTL_SECONDS = 3600
ALLOWED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt"}


@lru_cache
def get_llm() -> ChatOpenAI:
    return ChatOpenAI(model=LLM_MODEL, base_url=OPENAI_BASE_URL, temperature=0)


@lru_cache
def get_base_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, base_url=OPENAI_BASE_URL)
