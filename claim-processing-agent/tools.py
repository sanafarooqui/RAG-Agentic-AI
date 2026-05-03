import json
import os
from datetime import date

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from pydantic import ValidationError
from smolagents import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from models import ClaimsInfo, PolicyRecommendation, ClaimDecision
from rag import get_retriever

DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset", "coverage_data.csv")

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Shared state across the pipeline — tools write here so the LLM
# does not need to pass data between tool calls.
_state: dict = {}


def reset_state() -> None:
    _state.clear()


# ---------------------------------------------------------------------------
# Tool 1 — importClaims
# ---------------------------------------------------------------------------

@tool
def importClaims(claim_json_path: str) -> str:
    """
    Load a claim from a JSON file, validate required fields, and store it in
    the pipeline. Must be called first. Returns a status message.

    Args:
        claim_json_path: Path to the claim JSON file.
    """
    try:
        with open(claim_json_path, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        msg = f"[importClaims] FAILED — could not read file: {e}"
        print(msg)
        return msg

    try:
        claim = ClaimsInfo(**data)
    except ValidationError as e:
        errors = "; ".join(
            f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}"
            for err in e.errors()
        )
        msg = f"[importClaims] VALIDATION FAILED — {errors}"
        print(msg)
        return msg

    _state["claims_info"] = claim.model_dump(mode="json")
    msg = f"[importClaims] SUCCESS — Claim {claim.claim_number} loaded. Proceed to validateClaims."
    print(msg)
    return msg


# ---------------------------------------------------------------------------
# Tool 2 — validateClaims
# ---------------------------------------------------------------------------

@tool
def validateClaims() -> str:
    """
    Validate the imported claim against the coverage CSV. Checks policy
    existence, date of loss within coverage period, and no outstanding dues.
    Call after importClaims. Returns a status message.
    """
    claims_info = _state.get("claims_info")
    if not claims_info:
        msg = "[validateClaims] FAILED — no claim loaded. Call importClaims first."
        print(msg)
        return msg

    claim = ClaimsInfo(**claims_info)

    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        msg = f"[validateClaims] FAILED — dataset not found at {DATASET_PATH}"
        print(msg)
        return msg

    row = df[df["policy_number"] == claim.policy_number]
    if row.empty:
        msg = f"[validateClaims] FAILED — policy {claim.policy_number} not found."
        print(msg)
        return msg

    row = row.iloc[0]
    coverage_start = date.fromisoformat(str(row["coverage_start_date"]))
    coverage_end = date.fromisoformat(str(row["coverage_end_date"]))

    if not (coverage_start <= claim.date_of_loss <= coverage_end):
        msg = (
            f"[validateClaims] FAILED — date of loss {claim.date_of_loss} is outside "
            f"coverage period {coverage_start} to {coverage_end}."
        )
        print(msg)
        return msg

    if float(row["premium_dues_remaining"]) > 0:
        msg = (
            f"[validateClaims] FAILED — outstanding premium dues of "
            f"{row['premium_dues_remaining']} on policy {claim.policy_number}."
        )
        print(msg)
        return msg

    _state["validated"] = True
    msg = f"[validateClaims] SUCCESS — Policy {claim.policy_number} validated. Proceed to getRelevantPolicyInfo."
    print(msg)
    return msg


# ---------------------------------------------------------------------------
# Tool 3 — getRelevantPolicyInfo
# ---------------------------------------------------------------------------

_QUERY_GEN_PROMPT = ChatPromptTemplate.from_template(
    "You are an insurance expert. Based on the loss description below, generate "
    "2-3 concise search queries to retrieve the most relevant sections from a "
    "policy document. Return only the queries, one per line, no numbering.\n\n"
    "Loss description: {loss_description}"
)


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_fixed(2),
    reraise=True,
)
def _generate_queries(loss_description: str) -> list[str]:
    chain = _QUERY_GEN_PROMPT | _llm | StrOutputParser()
    raw = chain.invoke({"loss_description": loss_description})
    queries = [q.strip() for q in raw.strip().splitlines() if q.strip()]
    if not queries:
        raise ValueError("LLM returned no queries.")
    return queries


@tool
def getRelevantPolicyInfo() -> str:
    """
    Generate search queries from the claim's loss description and retrieve
    matching policy sections from ChromaDB. Call after validateClaims.
    Returns a status message.
    """
    claims_info = _state.get("claims_info")
    if not claims_info:
        msg = "[getRelevantPolicyInfo] FAILED — no claim loaded. Call importClaims first."
        print(msg)
        return msg

    claim = ClaimsInfo(**claims_info)

    try:
        queries = _generate_queries(claim.loss_description)
    except Exception as e:
        msg = f"[getRelevantPolicyInfo] FAILED — error generating queries: {e}"
        print(msg)
        return msg

    print(f"[getRelevantPolicyInfo] Queries: {queries}")

    retriever = get_retriever(k=5)
    seen, sections = set(), []
    for query in queries:
        try:
            for doc in retriever.invoke(query):
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    sections.append(doc.page_content)
        except Exception as e:
            print(f"[getRelevantPolicyInfo] Retrieval error for '{query}': {e}")

    if not sections:
        msg = "[getRelevantPolicyInfo] FAILED — no relevant policy sections found."
        print(msg)
        return msg

    _state["policy_context"] = "\n\n---\n\n".join(sections)
    msg = "[getRelevantPolicyInfo] SUCCESS — Policy sections retrieved. Proceed to processClaim."
    print(msg)
    return msg


# ---------------------------------------------------------------------------
# Tool 4 — processClaim
# ---------------------------------------------------------------------------

_PROCESS_PROMPT = ChatPromptTemplate.from_template(
    "You are an insurance claims adjuster. Analyze the claim and the relevant "
    "policy sections below and determine coverage applicability.\n\n"
    "CLAIM:\n{claim_json}\n\n"
    "POLICY SECTIONS:\n{policy_context}\n\n"
    "Respond with a JSON object only — no markdown, no extra text:\n"
    '{{"policy_section": "...", "recommendation_summary": "...", '
    '"deductible": <number>, "settlement_amount": <number>}}'
)


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_fixed(2),
    reraise=True,
)
def _call_process_llm(claim_json: str, policy_context: str) -> dict:
    chain = _PROCESS_PROMPT | _llm | StrOutputParser()
    raw = chain.invoke({"claim_json": claim_json, "policy_context": policy_context})
    parsed = json.loads(raw)
    return PolicyRecommendation(**parsed).model_dump()


@tool
def processClaim() -> str:
    """
    Analyze the imported claim against the retrieved policy sections and store
    a structured coverage recommendation. Call after getRelevantPolicyInfo.
    Returns a status message.
    """
    claims_info = _state.get("claims_info")
    policy_context = _state.get("policy_context")

    if not claims_info or not policy_context:
        msg = "[processClaim] FAILED — run importClaims and getRelevantPolicyInfo first."
        print(msg)
        return msg

    claim = ClaimsInfo(**claims_info)

    try:
        result = _call_process_llm(
            claim_json=json.dumps(claims_info, default=str),
            policy_context=policy_context,
        )
        _state["recommendation"] = result
        msg = f"[processClaim] SUCCESS — Recommendation ready for claim {claim.claim_number}. Proceed to finalDecision."
        print(msg)
        return msg
    except Exception as e:
        msg = f"[processClaim] FAILED after retries: {e}"
        print(msg)
        return msg


# ---------------------------------------------------------------------------
# Tool 5 — finalDecision
# ---------------------------------------------------------------------------

_DECISION_PROMPT = ChatPromptTemplate.from_template(
    "You are a senior insurance claims manager. Given the policy recommendation "
    "and the original claim, make a final coverage decision.\n\n"
    "CLAIM:\n{claim_json}\n\n"
    "RECOMMENDATION:\n{recommendation_json}\n\n"
    "Respond with a JSON object only — no markdown, no extra text:\n"
    '{{"claim_number": "...", "covered": <true|false>, "deductible": <number>, '
    '"recommended_payout": <number>, "notes": "..."}}'
)


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_fixed(2),
    reraise=True,
)
def _call_decision_llm(claim_json: str, recommendation_json: str) -> dict:
    chain = _DECISION_PROMPT | _llm | StrOutputParser()
    raw = chain.invoke({"claim_json": claim_json, "recommendation_json": recommendation_json})
    parsed = json.loads(raw)
    return ClaimDecision(**parsed).model_dump()


@tool
def finalDecision() -> str:
    """
    Apply the stored recommendation to produce and print the final claim
    decision. Call after processClaim. Returns a status message.
    """
    recommendation = _state.get("recommendation")
    claims_info = _state.get("claims_info")

    if not recommendation or not claims_info:
        msg = "[finalDecision] FAILED — run processClaim first."
        print(msg)
        return msg

    try:
        result = _call_decision_llm(
            claim_json=json.dumps(claims_info, default=str),
            recommendation_json=json.dumps(recommendation),
        )
        print("\n" + "=" * 55)
        print("  CLAIM DECISION")
        print("=" * 55)
        for key, value in result.items():
            print(f"  {key:<22}: {value}")
        print("=" * 55)
        _state["decision"] = result
        return "[finalDecision] SUCCESS — Claim processing complete."
    except Exception as e:
        msg = f"[finalDecision] FAILED after retries: {e}"
        print(msg)
        return msg
