import os
import sys

from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
os.environ["ANONYMIZED_TELEMETRY"] = "false"

from smolagents import ToolCallingAgent
from smolagents.models import OpenAIServerModel

from rag import ingest, CHROMA_DIR
from tools import importClaims, validateClaims, getRelevantPolicyInfo, processClaim, finalDecision, reset_state
from prompts import SYSTEM_PROMPT, PLANNING_PROMPT


def build_agent() -> ToolCallingAgent:
    model = OpenAIServerModel(
        model_id="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base=os.getenv("OPENAI_API_BASE"),
    )
    agent = ToolCallingAgent(
        tools=[importClaims, validateClaims, getRelevantPolicyInfo, processClaim, finalDecision],
        model=model,
    )
    agent.prompt_templates["system_prompt"] = SYSTEM_PROMPT
    return agent


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <path-to-claim.json>")
        print("Example: python main.py claims/sample_claim.json")
        sys.exit(1)

    claim_path = os.path.abspath(sys.argv[1])

    if not os.path.exists(claim_path):
        print(f"ERROR: Claim file not found: {claim_path}")
        sys.exit(1)

    if not os.path.exists(CHROMA_DIR):
        ingest()

    reset_state()
    agent = build_agent()
    task = PLANNING_PROMPT.format(claim_path=claim_path)
    agent.run(task, max_steps=10)


if __name__ == "__main__":
    main()
