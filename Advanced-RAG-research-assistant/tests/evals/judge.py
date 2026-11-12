import os

from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env"))
# The app's endpoint is an OpenAI-compatible proxy; the openai SDK used by DeepEval reads OPENAI_BASE_URL
if os.getenv("OPENAI_API_BASE_URL"):
    os.environ.setdefault("OPENAI_BASE_URL", os.environ["OPENAI_API_BASE_URL"])
# Evals must not add runs to the user's LangSmith project alongside their own chats
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "false"

from deepeval.models import GPTModel  # noqa: E402

JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "gpt-4o")


def get_judge() -> GPTModel:
    return GPTModel(model=JUDGE_MODEL, temperature=0)
