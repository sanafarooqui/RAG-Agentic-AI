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
