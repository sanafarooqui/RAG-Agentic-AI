# Claim Processing Agent

An autonomous insurance claim processing agent built with **SmolAgents**, **LangChain**, and **ChromaDB**. The agent ingests a policy PDF into a RAG pipeline, then processes a claim end-to-end — validating coverage, retrieving relevant policy sections, and producing a structured claim decision.

## Architecture

```
claim-processing-agent/
├── policy_docs/policy.pdf          # Source policy document
├── dataset/coverage_data.csv       # Policy coverage and dues data
├── claims/sample_claim.json        # Example claim input
├── models.py                       # Pydantic models (ClaimsInfo, PolicyRecommendation, ClaimDecision)
├── rag.py                          # PDF ingestion + ChromaDB retrieval
├── tools.py                        # 5 SmolAgent tools (pipeline steps)
├── prompts.py                      # System and planning prompts
├── main.py                         # Entry point
└── tests/                          # Pytest test suite (50 tests)
```

## Workflow

The agent calls 5 tools in sequence. Tools share state internally — no argument passing required between steps.

```
importClaims → validateClaims → getRelevantPolicyInfo → processClaim → finalDecision
```

| Step | Tool | What it does |
|---|---|---|
| 1 | `importClaims` | Loads claim JSON, validates required fields and date constraints |
| 2 | `validateClaims` | Checks policy exists in CSV, date of loss in coverage window, no outstanding dues |
| 3 | `getRelevantPolicyInfo` | LLM generates 2-3 search queries → retrieves matching policy sections from Chroma |
| 4 | `processClaim` | LLM analyzes claim vs policy → produces `PolicyRecommendation` |
| 5 | `finalDecision` | LLM produces final `ClaimDecision` and prints it to console |

## Setup

```bash
# From repo root — uses the shared venv
source .venv/bin/activate

# Fill in API credentials
cd claim-processing-agent
cp .env .env.local   # edit OPENAI_API_KEY and OPENAI_API_BASE
```

**`claim-processing-agent/.env`**
```
OPENAI_API_KEY=your-key-here
OPENAI_API_BASE=https://api.openai.com/v1
```

## Running

```bash
cd claim-processing-agent

# First run — ingests policy.pdf into ChromaDB, then processes the claim
python main.py claims/sample_claim.json

# Subsequent runs skip ingestion (chroma_db/ already exists)
python main.py claims/your_claim.json

# Re-ingest (e.g. after updating policy.pdf)
rm -rf chroma_db/
python main.py claims/sample_claim.json
```

## Claim JSON Format

```json
{
  "claim_number": "CLAIM-00001",
  "policy_number": "PN-1",
  "claimant_name": "Ema Johnson",
  "date_of_loss": "2022-01-21",
  "loss_description": "The car was rear ended by a truck when parked at the office.",
  "estimated_repair_cost": 550.0,
  "vehicle_details": "2022 Honda City"
}
```

**Validation rules:** `claim_number`, `policy_number`, and `loss_description` are required. `date_of_loss` must be today or in the past.

## Output

```
===========================================================
  CLAIM DECISION
===========================================================
  claim_number          : CLAIM-00001
  covered               : True
  deductible            : 500.0
  recommended_payout    : 50.0
  notes                 : Covered under Section 3 — Collision coverage.
===========================================================
```

## Dataset Format

`dataset/coverage_data.csv` must contain:

| Column | Description |
|---|---|
| `policy_number` | Unique policy identifier |
| `premium_dues_remaining` | Outstanding dues (0.0 = no dues) |
| `coverage_start_date` | ISO date (YYYY-MM-DD) |
| `coverage_end_date` | ISO date (YYYY-MM-DD) |

## Tests

```bash
/path/to/.venv/bin/pytest tests/ -v
```

50 tests covering happy paths, edge cases (coverage boundary dates), and error conditions (missing files, LLM failures, invalid data) for all 5 tools and `main.py`.
