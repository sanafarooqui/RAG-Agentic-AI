import os
import sys

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
os.environ["ANONYMIZED_TELEMETRY"] = "false"

from main import query, retrieve

judge = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class EvalScore(BaseModel):
    score: int = Field(description="Score from 1 to 5")
    reasoning: str = Field(description="One-sentence explanation of the score")


GROUNDEDNESS_PROMPT = ChatPromptTemplate.from_template(
    "You are an expert evaluator. Assess whether the ANSWER is fully supported by the CONTEXT.\n\n"
    "Groundedness measures if every claim in the answer can be traced back to the context, "
    "with no hallucinated or fabricated information.\n\n"
    "CONTEXT:\n{context}\n\n"
    "ANSWER:\n{answer}\n\n"
    "Score 1-5:\n"
    "1 - Answer contains major claims absent from the context\n"
    "2 - Answer mostly relies on information outside the context\n"
    "3 - Answer is partially grounded, some unsupported claims\n"
    "4 - Answer is mostly grounded with minor unsupported details\n"
    "5 - Every claim in the answer is fully supported by the context"
)

RELEVANCE_PROMPT = ChatPromptTemplate.from_template(
    "You are an expert evaluator. Assess whether the ANSWER directly addresses the QUESTION.\n\n"
    "Relevance measures how well the answer responds to what was actually asked, "
    "regardless of whether the information is correct.\n\n"
    "QUESTION:\n{question}\n\n"
    "ANSWER:\n{answer}\n\n"
    "Score 1-5:\n"
    "1 - Answer is completely off-topic\n"
    "2 - Answer barely addresses the question\n"
    "3 - Answer partially addresses the question\n"
    "4 - Answer mostly addresses the question with minor gaps\n"
    "5 - Answer directly and completely addresses the question"
)


def evaluate_groundedness(context: str, answer: str) -> EvalScore:
    chain = GROUNDEDNESS_PROMPT | judge.with_structured_output(EvalScore)
    return chain.invoke({"context": context, "answer": answer})


def evaluate_relevance(question: str, answer: str) -> EvalScore:
    chain = RELEVANCE_PROMPT | judge.with_structured_output(EvalScore)
    return chain.invoke({"question": question, "answer": answer})


def evaluate(question: str) -> dict:
    context = retrieve(question)
    answer = query(question)

    groundedness = evaluate_groundedness(context, answer)
    relevance = evaluate_relevance(question, answer)

    return {
        "question": question,
        "answer": answer,
        "groundedness": {"score": groundedness.score, "reasoning": groundedness.reasoning},
        "relevance": {"score": relevance.score, "reasoning": relevance.reasoning},
    }


def evaluate_batch(questions: list[str]) -> list[dict]:
    results = []
    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {question}")
        result = evaluate(question)
        results.append(result)
        print(f"  Answer      : {result['answer'][:120]}...")
        print(f"  Groundedness: {result['groundedness']['score']}/5 — {result['groundedness']['reasoning']}")
        print(f"  Relevance   : {result['relevance']['score']}/5 — {result['relevance']['reasoning']}")

    avg_g = sum(r["groundedness"]["score"] for r in results) / len(results)
    avg_r = sum(r["relevance"]["score"] for r in results) / len(results)

    print("\n" + "=" * 50)
    print(f"  Avg Groundedness : {avg_g:.1f} / 5")
    print(f"  Avg Relevance    : {avg_r:.1f} / 5")
    print("=" * 50)
    return results


if __name__ == "__main__":
    questions = sys.argv[1:] if len(sys.argv) > 1 else [
        "What is agentic AI?",
        "What are the main challenges in multi-agent systems?",
        "How do RAG systems improve LLM performance?",
        "What role does memory play in agentic AI systems?",
        "What are the safety concerns with autonomous AI agents?",
    ]
    evaluate_batch(questions)
