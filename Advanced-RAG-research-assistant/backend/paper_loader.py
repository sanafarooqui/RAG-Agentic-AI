import hashlib
import re
from bisect import bisect_right
from pathlib import Path

import arxiv
import httpx
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend import vector_store
from backend.config import ALLOWED_EXTENSIONS, ARXIV_DIR, CHUNK_OVERLAP, CHUNK_SIZE
from backend.models import DocumentRecord

ARXIV_ID_RE = re.compile(r"^(arxiv:)?(\d{4}\.\d{4,5}|[a-z\-]+(\.[A-Z]{2})?/\d{7})(v\d+)?$", re.IGNORECASE)

# Markdown "# Heading", or a numbered section "3.1 Title" alone on a line. PyMuPDF often puts the
# number and the title on separate lines. Titles are letters only, so table rows like
# "91.7 Transformer (4 layers)" are not mistaken for headings.
_HEADING_LINE = r"(?:#{1,6}[ \t]+\S[^\n]*|\d{1,2}(?:\.\d){0,2}\.?[ \n][A-Z][A-Za-z\- :,&']{3,79})"
HEADING_RE = re.compile(rf"^{_HEADING_LINE}$", re.MULTILINE)

# Listed first so splits prefer section boundaries over paragraph or sentence boundaries
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=[rf"\n(?={_HEADING_LINE}\n)", r"\n\n", r"\n", " ", ""],
    is_separator_regex=True,
    add_start_index=True,
)


def load_pages(path: Path) -> list[Document]:
    ext = path.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext}'. Allowed: pdf, md, txt")
    if ext == ".pdf":
        return PyMuPDFLoader(str(path)).load()
    return TextLoader(str(path), encoding="utf-8", autodetect_encoding=True).load()


def build_chunks(path: Path, title: str, origin: str) -> tuple[str, list[Document]]:
    """Split the whole document, not page by page, so chunks can span page breaks."""
    pages = load_pages(path)
    text = "\n".join(p.page_content for p in pages)
    if not text.strip():
        raise ValueError("No extractable text found in the document")

    page_starts, offset = [], 0
    for p in pages:
        page_starts.append(offset)
        offset += len(p.page_content) + 1
    headings = [(m.start(), " ".join(m.group().lstrip("#").split())) for m in HEADING_RE.finditer(text)]
    heading_starts = [pos for pos, _ in headings]

    doc_id = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    chunks = _splitter.split_documents([Document(page_content=text)])
    for i, chunk in enumerate(chunks):
        body = chunk.page_content
        # Judge position by the first visible character, so a chunk that opens with a heading belongs to it
        start = chunk.metadata["start_index"] + len(body) - len(body.lstrip())
        page = pages[bisect_right(page_starts, start) - 1].metadata.get("page")
        h = bisect_right(heading_starts, start) - 1
        section = headings[h][1] if h >= 0 else ""

        # The heading travels with the chunk text so it is part of what gets embedded
        chunk.page_content = f"[{title} | {section}]\n{body}" if section else f"[{title}]\n{body}"
        chunk.metadata = {
            "doc_id": doc_id,
            "title": title,
            "origin": origin,
            "page": int(page) + 1 if page is not None else 0,
            "chunk_index": i,
            "section": section,
        }
    return doc_id, chunks


def ingest(path: Path, title: str, origin: str, arxiv_id: str | None = None) -> DocumentRecord:
    doc_id, chunks = build_chunks(path, title, origin)
    if not vector_store.has_document(doc_id):
        vector_store.add_chunks(doc_id, chunks)
    return DocumentRecord(doc_id=doc_id, title=title, origin=origin, n_chunks=len(chunks), arxiv_id=arxiv_id)


def normalize_arxiv_id(query: str) -> str | None:
    q = query.strip()
    q = re.sub(r"^https?://arxiv\.org/(abs|pdf)/", "", q).removesuffix(".pdf")
    return re.sub(r"^arxiv:", "", q, flags=re.IGNORECASE) if ARXIV_ID_RE.match(q) else None


def arxiv_search(query: str, max_results: int = 5) -> list[dict]:
    arxiv_id = normalize_arxiv_id(query)
    search = (
        arxiv.Search(id_list=[arxiv_id])
        if arxiv_id
        else arxiv.Search(query=f'ti:"{query.strip()}"', max_results=max_results)
    )
    results = list(arxiv.Client().results(search))
    if not results and not arxiv_id:
        results = list(arxiv.Client().results(arxiv.Search(query=query.strip(), max_results=max_results)))
    return [
        {
            "arxiv_id": r.get_short_id(),
            "title": r.title,
            "authors": [a.name for a in r.authors][:4],
            "summary": r.summary,
            "published": r.published.date().isoformat(),
        }
        for r in results
    ]


def import_arxiv(arxiv_id: str) -> DocumentRecord:
    result = next(arxiv.Client().results(arxiv.Search(id_list=[arxiv_id])), None)
    if result is None:
        raise ValueError(f"No ArXiv paper found for '{arxiv_id}'")
    filename = f"{result.get_short_id().replace('/', '_')}.pdf"
    path = ARXIV_DIR / filename
    with httpx.stream("GET", result.pdf_url, follow_redirects=True, timeout=60) as resp:
        resp.raise_for_status()
        with path.open("wb") as out:
            for block in resp.iter_bytes():
                out.write(block)
    return ingest(path, title=result.title, origin="arxiv", arxiv_id=result.get_short_id())
