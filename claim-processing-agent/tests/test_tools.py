import json
import os
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import tools
from tools import (
    _state,
    finalDecision,
    getRelevantPolicyInfo,
    importClaims,
    processClaim,
    reset_state,
    validateClaims,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_state():
    reset_state()
    yield
    reset_state()


@pytest.fixture
def valid_claim_data():
    return {
        "claim_number": "CLAIM-001",
        "policy_number": "PN-1",
        "claimant_name": "Test User",
        "date_of_loss": "2022-01-21",
        "loss_description": "Vehicle rear-ended in parking lot.",
        "estimated_repair_cost": 550.0,
        "vehicle_details": "2022 Honda City",
    }


@pytest.fixture
def valid_claim_file(tmp_path, valid_claim_data):
    f = tmp_path / "claim.json"
    f.write_text(json.dumps(valid_claim_data))
    return str(f)


@pytest.fixture
def valid_csv(tmp_path):
    csv = tmp_path / "coverage_data.csv"
    csv.write_text(
        "policy_number,premium_dues_remaining,coverage_start_date,coverage_end_date\n"
        "PN-1,0.0,2021-01-01,2023-12-31\n"
        "PN-2,100.0,2021-01-01,2023-12-31\n"
    )
    return str(csv)


@pytest.fixture
def recommendation():
    return {
        "policy_section": "Section 3 — Collision",
        "recommendation_summary": "Covered under collision.",
        "deductible": 500.0,
        "settlement_amount": 50.0,
    }


@pytest.fixture
def decision():
    return {
        "claim_number": "CLAIM-001",
        "covered": True,
        "deductible": 500.0,
        "recommended_payout": 50.0,
        "notes": "Approved under collision coverage.",
    }


# ---------------------------------------------------------------------------
# importClaims
# ---------------------------------------------------------------------------

class TestImportClaims:

    def test_happy_path_stores_state(self, valid_claim_file):
        result = importClaims(claim_json_path=valid_claim_file)
        assert "SUCCESS" in result
        assert "CLAIM-001" in result
        assert _state["claims_info"]["claim_number"] == "CLAIM-001"
        assert _state["claims_info"]["policy_number"] == "PN-1"

    def test_file_not_found(self):
        result = importClaims(claim_json_path="/nonexistent/claim.json")
        assert "FAILED" in result
        assert "claims_info" not in _state

    def test_invalid_json(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("not {{ valid json")
        result = importClaims(claim_json_path=str(bad))
        assert "FAILED" in result
        assert "claims_info" not in _state

    def test_missing_claim_number(self, tmp_path, valid_claim_data):
        del valid_claim_data["claim_number"]
        f = tmp_path / "c.json"
        f.write_text(json.dumps(valid_claim_data))
        result = importClaims(claim_json_path=str(f))
        assert "VALIDATION FAILED" in result
        assert "claims_info" not in _state

    def test_missing_policy_number(self, tmp_path, valid_claim_data):
        del valid_claim_data["policy_number"]
        f = tmp_path / "c.json"
        f.write_text(json.dumps(valid_claim_data))
        result = importClaims(claim_json_path=str(f))
        assert "VALIDATION FAILED" in result

    def test_missing_loss_description(self, tmp_path, valid_claim_data):
        del valid_claim_data["loss_description"]
        f = tmp_path / "c.json"
        f.write_text(json.dumps(valid_claim_data))
        result = importClaims(claim_json_path=str(f))
        assert "VALIDATION FAILED" in result

    def test_future_date_of_loss_rejected(self, tmp_path, valid_claim_data):
        valid_claim_data["date_of_loss"] = str(date.today() + timedelta(days=1))
        f = tmp_path / "c.json"
        f.write_text(json.dumps(valid_claim_data))
        result = importClaims(claim_json_path=str(f))
        assert "VALIDATION FAILED" in result
        assert "claims_info" not in _state

    def test_date_today_is_valid(self, tmp_path, valid_claim_data):
        valid_claim_data["date_of_loss"] = str(date.today())
        f = tmp_path / "c.json"
        f.write_text(json.dumps(valid_claim_data))
        result = importClaims(claim_json_path=str(f))
        assert "SUCCESS" in result

    def test_empty_json_object(self, tmp_path):
        f = tmp_path / "empty.json"
        f.write_text("{}")
        result = importClaims(claim_json_path=str(f))
        assert "VALIDATION FAILED" in result

    def test_previous_state_cleared_on_new_import(self, tmp_path, valid_claim_data):
        _state["claims_info"] = {"claim_number": "OLD"}
        f = tmp_path / "c.json"
        f.write_text(json.dumps(valid_claim_data))
        importClaims(claim_json_path=str(f))
        assert _state["claims_info"]["claim_number"] == "CLAIM-001"


# ---------------------------------------------------------------------------
# validateClaims
# ---------------------------------------------------------------------------

class TestValidateClaims:

    def _seed(self, valid_claim_data, override=None):
        data = {**valid_claim_data, **(override or {})}
        _state["claims_info"] = data

    def test_happy_path(self, valid_claim_data, valid_csv):
        self._seed(valid_claim_data)
        with patch("tools.DATASET_PATH", valid_csv):
            result = validateClaims()
        assert "SUCCESS" in result
        assert _state.get("validated") is True

    def test_no_claim_in_state(self):
        result = validateClaims()
        assert "FAILED" in result
        assert "importClaims first" in result

    def test_policy_not_found(self, valid_claim_data, valid_csv):
        self._seed(valid_claim_data, {"policy_number": "PN-UNKNOWN"})
        with patch("tools.DATASET_PATH", valid_csv):
            result = validateClaims()
        assert "FAILED" in result
        assert "not found" in result

    def test_date_before_coverage_start(self, valid_claim_data, valid_csv):
        self._seed(valid_claim_data, {"date_of_loss": "2020-12-31"})
        with patch("tools.DATASET_PATH", valid_csv):
            result = validateClaims()
        assert "FAILED" in result
        assert "outside coverage period" in result

    def test_date_after_coverage_end(self, valid_claim_data, valid_csv):
        self._seed(valid_claim_data, {"date_of_loss": "2024-01-01"})
        with patch("tools.DATASET_PATH", valid_csv):
            result = validateClaims()
        assert "FAILED" in result
        assert "outside coverage period" in result

    def test_date_on_coverage_start_boundary(self, valid_claim_data, valid_csv):
        self._seed(valid_claim_data, {"date_of_loss": "2021-01-01"})
        with patch("tools.DATASET_PATH", valid_csv):
            result = validateClaims()
        assert "SUCCESS" in result

    def test_date_on_coverage_end_boundary(self, valid_claim_data, valid_csv):
        self._seed(valid_claim_data, {"date_of_loss": "2023-12-31"})
        with patch("tools.DATASET_PATH", valid_csv):
            result = validateClaims()
        assert "SUCCESS" in result

    def test_outstanding_premium_dues(self, valid_claim_data, valid_csv):
        self._seed(valid_claim_data, {"policy_number": "PN-2"})
        with patch("tools.DATASET_PATH", valid_csv):
            result = validateClaims()
        assert "FAILED" in result
        assert "premium dues" in result

    def test_dataset_file_not_found(self, valid_claim_data):
        self._seed(valid_claim_data)
        with patch("tools.DATASET_PATH", "/nonexistent/coverage.csv"):
            result = validateClaims()
        assert "FAILED" in result
        assert "dataset not found" in result

    def test_zero_dues_passes(self, valid_claim_data, valid_csv):
        self._seed(valid_claim_data, {"policy_number": "PN-1"})
        with patch("tools.DATASET_PATH", valid_csv):
            result = validateClaims()
        assert "SUCCESS" in result


# ---------------------------------------------------------------------------
# getRelevantPolicyInfo
# ---------------------------------------------------------------------------

class TestGetRelevantPolicyInfo:

    def _seed(self, valid_claim_data):
        _state["claims_info"] = valid_claim_data

    def _make_doc(self, content):
        doc = MagicMock()
        doc.page_content = content
        return doc

    def test_happy_path(self, valid_claim_data):
        self._seed(valid_claim_data)
        doc = self._make_doc("Section 3: Collision coverage applies.")
        with (
            patch("tools._generate_queries", return_value=["rear-end collision"]),
            patch("tools.get_retriever") as mock_retriever,
        ):
            mock_retriever.return_value.invoke.return_value = [doc]
            result = getRelevantPolicyInfo()
        assert "SUCCESS" in result
        assert "policy_context" in _state
        assert "Section 3" in _state["policy_context"]

    def test_no_claim_in_state(self):
        result = getRelevantPolicyInfo()
        assert "FAILED" in result
        assert "importClaims first" in result
        assert "policy_context" not in _state

    def test_query_generation_fails(self, valid_claim_data):
        self._seed(valid_claim_data)
        with patch("tools._generate_queries", side_effect=Exception("LLM timeout")):
            result = getRelevantPolicyInfo()
        assert "FAILED" in result
        assert "policy_context" not in _state

    def test_empty_retrieval_results(self, valid_claim_data):
        self._seed(valid_claim_data)
        with (
            patch("tools._generate_queries", return_value=["query"]),
            patch("tools.get_retriever") as mock_retriever,
        ):
            mock_retriever.return_value.invoke.return_value = []
            result = getRelevantPolicyInfo()
        assert "FAILED" in result
        assert "no relevant policy sections" in result

    def test_duplicate_docs_are_deduplicated(self, valid_claim_data):
        self._seed(valid_claim_data)
        doc = self._make_doc("Repeated content.")
        with (
            patch("tools._generate_queries", return_value=["q1", "q2"]),
            patch("tools.get_retriever") as mock_retriever,
        ):
            mock_retriever.return_value.invoke.return_value = [doc, doc]
            result = getRelevantPolicyInfo()
        assert "SUCCESS" in result
        assert _state["policy_context"].count("Repeated content.") == 1

    def test_retriever_error_on_one_query_continues(self, valid_claim_data):
        self._seed(valid_claim_data)
        doc = self._make_doc("Good section.")
        retriever = MagicMock()
        retriever.invoke.side_effect = [Exception("Chroma error"), [doc]]
        with (
            patch("tools._generate_queries", return_value=["q1", "q2"]),
            patch("tools.get_retriever", return_value=retriever),
        ):
            result = getRelevantPolicyInfo()
        assert "SUCCESS" in result
        assert "Good section." in _state["policy_context"]

    def test_multiple_queries_combine_unique_sections(self, valid_claim_data):
        self._seed(valid_claim_data)
        doc_a = self._make_doc("Section A.")
        doc_b = self._make_doc("Section B.")
        retriever = MagicMock()
        retriever.invoke.side_effect = [[doc_a], [doc_b]]
        with (
            patch("tools._generate_queries", return_value=["q1", "q2"]),
            patch("tools.get_retriever", return_value=retriever),
        ):
            getRelevantPolicyInfo()
        assert "Section A." in _state["policy_context"]
        assert "Section B." in _state["policy_context"]


# ---------------------------------------------------------------------------
# processClaim
# ---------------------------------------------------------------------------

class TestProcessClaim:

    def _seed(self, valid_claim_data):
        _state["claims_info"] = valid_claim_data
        _state["policy_context"] = "Section 3: Collision coverage."

    def test_happy_path(self, valid_claim_data, recommendation):
        self._seed(valid_claim_data)
        with patch("tools._call_process_llm", return_value=recommendation):
            result = processClaim()
        assert "SUCCESS" in result
        assert _state["recommendation"] == recommendation

    def test_no_claims_info(self):
        _state["policy_context"] = "Some context."
        result = processClaim()
        assert "FAILED" in result
        assert "recommendation" not in _state

    def test_no_policy_context(self, valid_claim_data):
        _state["claims_info"] = valid_claim_data
        result = processClaim()
        assert "FAILED" in result
        assert "recommendation" not in _state

    def test_both_missing(self):
        result = processClaim()
        assert "FAILED" in result

    def test_llm_failure_returns_failed(self, valid_claim_data):
        self._seed(valid_claim_data)
        with patch("tools._call_process_llm", side_effect=Exception("LLM down")):
            result = processClaim()
        assert "FAILED" in result
        assert "recommendation" not in _state

    def test_claim_number_in_success_message(self, valid_claim_data, recommendation):
        self._seed(valid_claim_data)
        with patch("tools._call_process_llm", return_value=recommendation):
            result = processClaim()
        assert "CLAIM-001" in result


# ---------------------------------------------------------------------------
# finalDecision
# ---------------------------------------------------------------------------

class TestFinalDecision:

    def _seed(self, valid_claim_data, recommendation):
        _state["claims_info"] = valid_claim_data
        _state["recommendation"] = recommendation

    def test_happy_path(self, valid_claim_data, recommendation, decision):
        self._seed(valid_claim_data, recommendation)
        with patch("tools._call_decision_llm", return_value=decision):
            result = finalDecision()
        assert "SUCCESS" in result
        assert _state["decision"] == decision

    def test_no_recommendation(self, valid_claim_data):
        _state["claims_info"] = valid_claim_data
        result = finalDecision()
        assert "FAILED" in result
        assert "decision" not in _state

    def test_no_claims_info(self, recommendation):
        _state["recommendation"] = recommendation
        result = finalDecision()
        assert "FAILED" in result

    def test_both_missing(self):
        result = finalDecision()
        assert "FAILED" in result

    def test_llm_failure_returns_failed(self, valid_claim_data, recommendation):
        self._seed(valid_claim_data, recommendation)
        with patch("tools._call_decision_llm", side_effect=Exception("timeout")):
            result = finalDecision()
        assert "FAILED" in result
        assert "decision" not in _state

    def test_decision_printed_to_console(self, valid_claim_data, recommendation, decision, capsys):
        self._seed(valid_claim_data, recommendation)
        with patch("tools._call_decision_llm", return_value=decision):
            finalDecision()
        out = capsys.readouterr().out
        assert "CLAIM DECISION" in out
        assert "CLAIM-001" in out

    def test_not_covered_decision(self, valid_claim_data, recommendation):
        self._seed(valid_claim_data, recommendation)
        not_covered = {
            "claim_number": "CLAIM-001",
            "covered": False,
            "deductible": 0.0,
            "recommended_payout": 0.0,
            "notes": "Excluded under policy Section 5.",
        }
        with patch("tools._call_decision_llm", return_value=not_covered):
            result = finalDecision()
        assert "SUCCESS" in result
        assert _state["decision"]["covered"] is False


# ---------------------------------------------------------------------------
# reset_state
# ---------------------------------------------------------------------------

class TestResetState:

    def test_clears_all_keys(self, valid_claim_data, recommendation):
        _state["claims_info"] = valid_claim_data
        _state["validated"] = True
        _state["policy_context"] = "context"
        _state["recommendation"] = recommendation
        _state["decision"] = {}
        reset_state()
        assert _state == {}
