# Autonomous Financial Research Analyst

A two-part JHU project that builds a goal-oriented, autonomous financial research agent using LangGraph and LangChain. The agent autonomously gathers public market data, searches real-time news, and queries private analyst documents to produce investment-grade research reports.

## Business Problem

Investment analysts spend 60–70% of their time on data collection and only 30–40% on actual analysis. This project reduces research time from 4–6 hours per company to 5–10 minutes by deploying an autonomous agent that proactively gathers and synthesizes both quantitative and qualitative signals.

## Project Structure

```
autonomous-financial-analyst/
├── Autonomous_financial_analyst_Learners_Notebook.ipynb   # Main notebook (Parts 1 & 2)
├── Autonomous_financial_analyst_Learners_Notebook.html    # HTML export for submission
├── Companies-AI-Initiatives.zip                           # Private analyst PDFs for RAG
└── config.json                                            # API keys (not committed)
```

## Part 1 — Defining the Agent's Charter

Covers the four characteristics of autonomous agents and compares a traditional reactive LLM against a fully chartered autonomous agent.

### Tools (Actuators)

| Tool | Description |
|------|-------------|
| `get_stock_price(ticker)` | Real-time price, volume, market cap via Yahoo Finance |
| `get_stock_history(ticker, period)` | Historical OHLCV data; use `"3y"` for 3-year analysis |
| `search_financial_news(query)` | Real-time news via Tavily Search API |
| `analyze_sentiment(text)` | Sentiment scoring via `gpt-4o-mini` |

### Agent Architecture (LangGraph)

```
User Query
    │
    ▼
Agent Node (LLM decides tool calls)
    │
    ├──► Tool Node (executes selected tools)
    │        │
    │        └──► returns results → Agent Node
    │
    └──► END (returns final report to user)
```

### Agent Charter Principles

- **Proactiveness** — always gathers 3-year history and recent news, even when not explicitly asked
- **Reactivity** — on tool failure, immediately tries an alternative approach and continues analysis
- **Autonomy** — independently decides tool order and combination
- **Transparency** — every factual claim is cited with `[Source: tool_name, timestamp]`; all recommendations include a confidence level

### Comparative Tests

1. Traditional reactive LLM — responds only to what is asked
2. Basic goal-oriented agent — proactively uses multiple tools
3. Full autonomous agent — uses all 4 tools, cites sources, assesses risk, gives Buy/Hold/Sell with confidence
4. Error handling scenario — simulates `get_stock_price` timeout; agent falls back to history and news

## Part 2 — RAG & Investment Intelligence

Extends the agent with a fifth tool backed by a RAG pipeline over private analyst PDF reports, then builds a multi-company investment ranking system.

### RAG Pipeline

| Step | Implementation |
|------|---------------|
| Document loading | `PyPDFDirectoryLoader` over `Companies-AI-Initiatives/` |
| Chunking | `RecursiveCharacterTextSplitter`, tiktoken `cl100k_base`, chunk_size=1000, overlap=200 |
| Embeddings | `text-embedding-ada-002` via `OpenAIEmbeddings` |
| Vector store | `ChromaDB`, collection `"AI_Initiatives"`, top-k=10 similarity retrieval |

### Fifth Tool

```python
query_private_database(query: str) -> str
```

Retrieves relevant passages from the private analyst reports and generates a grounded, source-cited answer.

### Updated Agent Charter (Part 2)

Adds a mandatory AI Research Activity Check: for every company the agent must query the private database and report:
- Is the company actively engaged in AI R&D? (Yes/No with evidence)
- Top 3 AI research areas or projects
- Strategic focus and commitment level

### Investment Ranking System

Analyzes five companies simultaneously and ranks them by two criteria:

| Criterion | Source |
|-----------|--------|
| Financial performance | 3-year stock return, current price, market cap |
| AI innovation | Active projects, research areas, strategic timelines from private docs |

**Companies analyzed:** MSFT, GOOGL, NVDA, AMZN, IBM

## Setup

### Prerequisites

```bash
pip install langchain==0.3.27 langchain-core==0.3.79 langchain-openai==0.3.35 \
  langchain-community==0.3.31 langgraph==0.6.6 tavily-python yfinance==0.2.66 \
  chromadb==1.3.4 pypdf==6.10.2 tiktoken==0.12.0
```

### API Keys

Create `config.json` in the project directory (not committed):

```json
{
  "API_KEY": "your_openai_api_key",
  "OPENAI_API_BASE": "https://your_openai_api_base/v1",
  "TAVILY_API_KEY": "your_tavily_api_key"
}
```

### Run

Open the notebook in Jupyter and execute cells sequentially. Part 1 cells are self-contained; Part 2 requires `Companies-AI-Initiatives.zip` to be present alongside the notebook.

## Key Takeaways

- **LangGraph state machine** cleanly separates agent reasoning from tool execution
- **RAG unlocks private knowledge** without retraining — document updates require only re-embedding
- **Synergistic tool usage**: stock data + news sentiment + RAG together produce insights no single tool could provide
- **Graceful degradation**: the agent never halts on a single tool failure — it adapts and reports the data gap
