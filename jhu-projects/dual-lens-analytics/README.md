# DualLens Analytics

A JHU Project 1 that blends quantitative financial analysis with qualitative AI strategy insights using Retrieval-Augmented Generation (RAG). Investors get a unified ranking of companies based on both stock performance and innovation readiness — the "dual lens."

## Business Problem

Traditional investment analysis treats financial metrics and strategic documents as separate worlds. Analysts must manually reconcile 3-year stock trends with PDF strategy reports, leading to fragmented insights, missed high-potential companies, and decisions that ignore innovation capacity.

**DualLens Analytics** addresses three core challenges:
1. **Fragmented data sources** — financial data and AI initiative PDFs exist in silos
2. **Limited analytical scope** — manual analysis of both is time-consuming and error-prone
3. **Decisional blind spots** — without combining quantitative and qualitative signals, high-potential companies are overlooked

## Project Structure

```
dual-lens-analytics/
├── JHU_AgenticAI_Project_1_Learners_Notebook (2).ipynb   # Main notebook
├── JHU_AgenticAI_Project_1_Learners_Notebook.html        # HTML export for submission
└── Companies-AI-Initiatives.zip                          # AI initiative PDFs for RAG
```

## Workflow Overview

```
1. Organization Selection  (5 companies)
         │
         ▼
2. LLM Setup              (gpt-4o-mini via ChatOpenAI)
         │
         ▼
3. Financial Analysis     (yfinance: 3-year stock trends + 5 financial metrics)
         │
         ▼
4. RAG Pipeline           (PDF load → chunk → embed → ChromaDB → semantic retrieval)
         │
         ▼
5. Q&A over Private Docs  (natural-language queries about AI initiatives)
         │
         ▼
6. LLM-as-Judge Evaluation (groundedness + relevance scoring)
```

## Companies Analyzed

`GOOGL`, `MSFT`, `IBM`, `NVDA`, `AMZN`

## Section Details

### 1. Financial Visualization (Section 3)

- Pulls 3-year closing price history via `yfinance` and plots all five companies on a single chart
- Fetches five financial metrics per company and renders individual bar charts:

| Metric | Description |
|--------|-------------|
| Market Cap | Total market value of outstanding shares (in $B) |
| P/E Ratio | Price investors pay per dollar of earnings |
| Dividend Yield | Annual dividend as % of stock price |
| Beta | Volatility relative to overall market |
| Total Revenue | Total business income (in $B) |

### 2. RAG Pipeline (Section 4)

**A. Document Loading**
- Extracts `Companies-AI-Initiatives.zip` and loads all PDFs with `PyPDFDirectoryLoader`

**B. Chunking & Embedding**
- Splits documents with `RecursiveCharacterTextSplitter` (tiktoken `cl100k_base`, chunk_size=600, overlap=75)
- Embeds chunks using `text-embedding-ada-002` via `OpenAIEmbeddings`
- Stores in `ChromaDB` collection `"AI_Initiatives"`, retriever set to top-k=10 similarity

**C. Retrieval & Q&A**

The `RAG(user_message)` function:
1. Retrieves the 10 most semantically relevant chunks from ChromaDB
2. Assembles a context string
3. Calls `gpt-4o-mini` with a structured system prompt that restricts the answer to the retrieved context

Sample queries run in the notebook:
- "How is the area in which GOOGL is working different from the area in which MSFT is working?"
- "What are the three projects on which MSFT is working upon?"
- "What is the timeline of each project in NVDA?"
- "What are the areas in which AMZN is investing when it comes to AI?"
- "What are the risks associated with projects within GOOG?"

**D. LLM-as-Judge Evaluation**

Evaluates RAG quality on two axes using a second `gpt-4o-mini` instance as judge (1–5 scale):

| Metric | What it measures |
|--------|-----------------|
| **Groundedness** | Answer is derived only from the retrieved context |
| **Relevance** | Retrieved context actually supports answering the question |

## Setup

### Prerequisites

```bash
pip install -U langchain langchain-core langchain-community langchain-openai \
  langchain-text-splitters pypdf chromadb yfinance matplotlib pandas
```

### API Keys

Create `config.json` in the project directory (not committed):

```json
{
  "API_KEY": "your_openai_api_key",
  "OPENAI_API_BASE": "https://your_openai_api_base/v1"
}
```

### Run

Open the notebook in Jupyter and execute all cells in order. The notebook was originally developed in Google Colab — file paths such as `/content/Companies-AI-Initiatives/` may need adjustment if running locally.

## Key Takeaways

- **RAG enables qualitative analysis at scale** — natural-language queries over private PDFs without manual reading
- **Dual-lens ranking** combines hard financial numbers with soft innovation signals for richer investment decisions
- **LLM-as-Judge** provides repeatable, automated quality control over RAG responses without human annotation
