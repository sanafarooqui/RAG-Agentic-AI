import os
import sys
import zipfile
import tempfile

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
os.environ["ANONYMIZED_TELEMETRY"] = "false"

ZIP_PATH = os.path.join(os.path.dirname(__file__), "agentic_ai_research_papers.zip")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "research_papers"

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=500,
)

embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')

RAG_PROMPT = ChatPromptTemplate.from_template(
    "You are a research assistant. Answer using only the context provided. "
    "If the answer is not in the context, say you don't know.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)


def extract_zip(zip_path: str) -> str:
    extract_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    return extract_dir


def load_and_split(pdf_dir: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    loader = PyPDFDirectoryLoader(pdf_dir)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)


def ingest():
    print(f"Extracting PDFs from {ZIP_PATH} ...")
    pdf_dir = extract_zip(ZIP_PATH)

    print("Loading and splitting documents ...")
    chunks = load_and_split(pdf_dir)
    print(f"  {len(chunks)} chunks created.")

    print("Building Chroma vector store ...")
    Chroma.from_documents(
        chunks,
        embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    print(f"Vector store persisted to {CHROMA_DIR}")


def _get_retriever():
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    return vector_store.as_retriever(search_kwargs={"k": 5})


def retrieve(question: str) -> str:
    docs = _get_retriever().invoke(question)
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain():
    retriever = _get_retriever()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )


def query(question: str) -> str:
    return build_rag_chain().invoke(question)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        ingest()
    else:
        if not os.path.exists(CHROMA_DIR):
            print("Vector store not found. Run first: python main.py ingest")
            sys.exit(1)
        question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Question: ")
        print(query(question))
