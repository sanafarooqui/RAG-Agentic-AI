import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

POLICY_PDF = os.path.join(os.path.dirname(__file__), "policy_docs", "policy.pdf")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "policy_docs"

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Policy PDFs have numbered sections and legal clauses.
# Split on double newlines first (section breaks), then single newlines,
# then sentence boundaries — keeping clauses intact within chunk_size.
_SPLITTER = RecursiveCharacterTextSplitter(
   # separators=["\n\n", "\n", ". ", " "],
    chunk_size=1000,
    chunk_overlap=200,
)


def ingest() -> None:
    print(f"Loading policy PDF from {POLICY_PDF} ...")
    loader = PyPDFLoader(POLICY_PDF)
    pages = loader.load()

    chunks = _SPLITTER.split_documents(pages)
    print(f"  {len(pages)} pages → {len(chunks)} chunks.")

    print("Building Chroma vector store ...")
    Chroma.from_documents(
        chunks,
        embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    print(f"Vector store persisted to {CHROMA_DIR}")


def get_retriever(k: int = 5):
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    return vector_store.as_retriever(search_kwargs={"k": k})
