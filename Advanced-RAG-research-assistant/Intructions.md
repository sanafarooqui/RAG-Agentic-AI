This project focuses on building, testing, evaluating an advanced RAG application.

RAG Research assistant 
The main application will be an agentic RAG chatbot for working with research papers. There will be 2 functionalities - 
1. Users will be able to upload documents in formats such as pdf, markdown and text and ask questions.
2. Users will be able to search a research paper on ArXiv by title or identification number and bring them into the research flow.
3.A chat will be provided as the main functionality where users can ask questions on a research paper.
The application should preserve conversation history and support multiple independent sessions.Short term memory will allow assistant to remember earlier messages.Each session can function as individual collection of question, answers, retrieved  evidence and selected documents.Users should be able to begin a new conversation through a "New button"

Agentic Routing and retrieval
A router will determine how each user query is handled.If a user query can be answered using uploaded research material it will be sent to knowledge base retriever.If local documents do not contain adequate information, the request can be routed to another source such as web search or models internal knowledge.
The retrieval process should contain quality controls.After retrieval a relevancy check will determine whether the passages address the user's question.If the retrieved information is sufficiently relevant, the system will generate a grounded response.If retrieval is weak , the system may rewrite the query and try again.A fallback mechanism should be included for cases in which suitable evidence cannot be found. Retrieved documents maybe reranked to place most useful passages first, reducing latency and improving answer quality

Web search will serve two important purposes: filling knowledge gaps and verifying claims. The system should distinguish between material supported by internal research documents and information obtained from external sources. Where possible, responses should identify their supporting resources so users can trace claims back to the original evidence. This is particularly important for a research assistant, where transparency and factual grounding matter as much as fluency.

**Local Development and Performance**

Development is planned in two broad phases. The first phase will produce a locally running demonstration, including the agentic RAG backend and a React frontend. This phase will establish the core workflow for document ingestion, retrieval, question answering, session management, and evaluation. The second phase will strengthen the RAG pipeline, complete formal evaluation, containerize the application, and deploy it to AWS.

Performance optimization will be important as the document collection grows. Computing embeddings repeatedly can be expensive and slow, so the system should cache previously generated embeddings. Each document chunk can be paired with its embedding vector and stored for reuse. A key-value store could maintain session and temporary data using memory, file storage, or Redis. Redis may be especially useful for larger workloads, such as collections exceeding ten million records. Caching the most recent sessions and limiting unnecessary API calls should reduce latency and operating costs.


**Evaluation Framework**

The evaluation phase will assess whether the RAG system retrieves appropriate evidence and produces accurate, relevant answers. The project notes identify RAGAS and DeepEval as possible evaluation frameworks. Core RAG metrics should include context precision, context recall, context relevancy, faithfulness, and response relevancy. Together, these metrics evaluate whether the retrieved context is useful, whether important evidence was missed, whether the answer is supported by that evidence, and whether the final response addresses the question.

Evaluation requires a structured dataset of question-answer pairs. Each evaluation record should contain an input query, the expected answer, the context retrieved by the RAG pipeline, and the answer produced by the language model. A collection of these records will form a set of test cases. The evaluation process can then run those cases through the pipeline, calculate the selected metrics, and generate a detailed report. An LLM-as-a-judge approach may be used, with a capable model scoring outputs according to clearly defined criteria.

Some evaluation data can be designed manually as "golden" question-answer pairs. These examples provide a trusted reference against which the system's output can be compared. DeepEval's synthesizer may also generate evaluation cases from source documents. The workflow would load documents, split them into chunks, generate embeddings, and store them in a vector database such as Chroma. Random chunks could then be sampled and filtered according to criteria such as clarity, depth, structure, and relevance.

A filtering threshold, such as 0.5 on a zero-to-one scale, can be applied to remove weak chunks. The retained chunks can then be grouped according to semantic similarity using embeddings. Related chunks may be merged into a larger context and passed to a language model, which generates a query and expected response grounded in that context. This produces synthetic golden cases that can supplement manually prepared RAGAS cases. A balanced evaluation dataset should combine both approaches: carefully designed human examples and systematically generated synthetic examples.

**Deployment**

The final application will be packaged in a Docker container. A Dockerfile will provide the instructions needed to assemble the application and its dependencies into a reproducible image. After the image is built, it will be pushed to a container registry or Docker Hub.

For AWS deployment, an EC2 instance will be created and configured with Docker. The server will authenticate with the container registry, pull the application image, and run it as a Docker container. This approach separates the application from the underlying server environment and makes deployment more consistent. Once the container is running and the required ports and security settings are configured, the RAG research assistant will be accessible as a deployed web application.

**Implementation instrustions** - 
Implement phase 1
The UI should have left navigation bar and a main chat panel.
Left bar should contain upload button to upload research document, chat history panel,an ArXiv search input field and a web search input box.
RAG pipeline will be implemented per - Local Development and Performance section.

Folder structure and code should be implemented similar to this project - https://github.com/Himanshu-1703/rag-papeer-project

**Phase 2 - Evaluation of the RAG**

We will perform 2 evaluations - 
1. Online evaluation (langsmith) - already setup
2.Offline evaluation using DeepEval. (Skill included)

For offline evluation should consist of
1. Evaluating RAG components individually
 Retriever - Recall and Precision
 Generator - Faithfulness,correctness, relevance,completeness any other essential metrics
 2. Evaluate the whole RAG pipeline - contexual relevance
 3. Evaluate the application as a whole for safety (toxicity, PII masking), operations and security

**VERY IMPORTANT** - After eval, tell me what the scores are, what tests failed reason and how we can fix it.

Read below instructions for getting started with DeepEval - 
I want to use DeepEval as my build-loop ground truth, not just a validation
step at the end. You — the coding agent — will run evals, read the failures
and traces, and use them as the source of truth for what to change next in
my AI app. Then re-run to confirm.

## DeepEval Resources

**Documentation:**
- Main docs: https://www.deepeval.com/docs
- 5-min Quickstart: https://www.deepeval.com/docs/getting-started
- Vibe Coding (the loop): https://www.deepeval.com/docs/vibe-coding
- Agents Quickstart: https://www.deepeval.com/docs/getting-started-agents
- Voice Agents: https://www.deepeval.com/docs/evaluation-voice
- Chatbot Quickstart: https://www.deepeval.com/docs/getting-started-chatbots
- RAG Quickstart: https://www.deepeval.com/docs/getting-started-rag
- Metrics catalog: https://www.deepeval.com/docs/metrics-introduction
- CLI reference: https://www.deepeval.com/docs/command-line-interface
- LLM-friendly docs: https://www.deepeval.com/llms.txt

**Integrations (use these when applicable — see "Framework Integrations First" below):**
- Integrations index: https://www.deepeval.com/integrations
- OpenAI Agents SDK: https://www.deepeval.com/integrations/frameworks/openai-agents
- OpenAI SDK: https://www.deepeval.com/integrations/frameworks/openai
- Anthropic SDK: https://www.deepeval.com/integrations/frameworks/anthropic
- LangChain: https://www.deepeval.com/integrations/frameworks/langchain
- LangGraph: https://www.deepeval.com/integrations/frameworks/langgraph
- LlamaIndex: https://www.deepeval.com/integrations/frameworks/llamaindex
- CrewAI: https://www.deepeval.com/integrations/frameworks/crewai
- PydanticAI: https://www.deepeval.com/integrations/frameworks/pydanticai
- Voice agent connectors (Vapi, ElevenLabs, LiveKit, Pipecat): https://www.deepeval.com/integrations#voice-agent-connectors
- Google ADK: https://www.deepeval.com/integrations/frameworks/google-adk
- AWS AgentCore: https://www.deepeval.com/integrations/frameworks/agentcore
- HuggingFace: https://www.deepeval.com/integrations/frameworks/huggingface

**Code & Skill:**
- Core repo: https://github.com/confident-ai/deepeval
- Python SDK: pip install -U deepeval
- Agent Skill (carries the iteration loop): npx skills add confident-ai/deepeval --skill deepeval

## Framework Integrations First (IMPORTANT)

Before adding ANY tracing code, detect whether my app already uses one of the
supported frameworks above. If it does, **use the DeepEval integration for that
framework instead of manually instrumenting with `@observe`**. Integrations
auto-instrument every agent/chain run, every LLM call, and every tool call —
producing the same trace + span structure DeepEval evaluates against, with
zero hand-written decorators.

Detection cheat sheet (check `pyproject.toml`, `requirements.txt`, and imports):
- `openai-agents` / `from agents import Agent` → OpenAI Agents SDK integration
- `openai` (without `agents`) → OpenAI SDK integration
- `anthropic` → Anthropic SDK integration
- `langchain` / `langchain-*` → LangChain integration
- `langgraph` → LangGraph integration
- `llama-index` → LlamaIndex integration
- `crewai` → CrewAI integration
- `pydantic-ai` → PydanticAI integration
- `google-adk` → Google ADK integration
- AWS AgentCore agents → AgentCore integration
- HuggingFace `transformers` / `smolagents` → HuggingFace integration

If a matching integration exists, fetch its docs page (URL above) and follow
its instrumentation pattern verbatim — typically a single `instrument=...`
argument, a `Settings(...)` object, or one wrapper call at app construction
time. Do not also add `@observe` over the same code paths; the integration
already produces those spans.

Only fall back to manual `@observe` instrumentation when:
- The app uses a framework with no DeepEval integration, OR
- The app is plain Python with no framework, OR
- The user explicitly asks for hand-rolled tracing.

## How DeepEval Plugs Into Your Loop

- Test cases (LLMTestCase / ConversationalTestCase) describe one behavior.
- Goldens are dataset entries the agent app is invoked on.
- Metrics score test cases and return: score (0–1), pass/fail vs threshold,
  and a natural-language `reason` you can read.
- Framework integrations (preferred) auto-instrument the app so every
  agent run, LLM call, and tool call becomes an evaluable span.
- `@observe` (fallback) traces the app manually when no integration applies.
- `deepeval test run` runs the suite and prints per-metric, per-span results
  you can parse without an explicit "summarize this" step.
- `deepeval generate` synthesizes goldens from docs, contexts, or scratch
  when no dataset exists yet.

## Your Job (the Build Loop)

For each iteration round:
  1. Run `deepeval test run tests/evals/test_<app>.py`.
  2. Read the per-metric scores and `reason` strings. Identify the
     lowest-scoring metric and the spans/test cases that caused it.
  3. Pick the smallest likely app change — prompt, retrieval scoping,
     tool wiring, parser, instructions. Do NOT edit the metric, lower
     the threshold, or delete failing goldens.
  4. Edit the app code. Keep the change scoped.
  5. Re-run the eval suite. Confirm the failing metric improved
     without regressing other metrics.
  6. Summarize: what failed, what you changed, what moved.
Repeat for the requested number of rounds (default 5).

## Start Here

1. Detect the framework (see "Framework Integrations First" above) and tell
   me which integration you'll use, OR confirm there's no match and you'll
   fall back to manual `@observe`.
2. Ask me what I'm building (agent / voice agent / chatbot / RAG / plain LLM), what
   dataset I have (or whether to generate one with `deepeval generate`),
   and whether I want results pushed to Confident AI.
3. Set up a committed pytest eval suite under `tests/evals/`, do one round
   of the loop end-to-end, and only then ask me what to focus on next.