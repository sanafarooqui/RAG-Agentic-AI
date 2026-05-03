import os
import sys
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_main(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", argv)
    from main import main
    main()


# ---------------------------------------------------------------------------
# CLI argument handling
# ---------------------------------------------------------------------------

class TestCLI:

    def test_no_args_prints_usage_and_exits(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["main.py"])
        with pytest.raises(SystemExit) as exc:
            from main import main
            main()
        assert exc.value.code == 1
        assert "Usage" in capsys.readouterr().out

    def test_missing_claim_file_exits(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["main.py", "/nonexistent/claim.json"])
        with pytest.raises(SystemExit) as exc:
            from main import main
            main()
        assert exc.value.code == 1
        assert "not found" in capsys.readouterr().out

    def test_valid_claim_file_proceeds(self, tmp_path, monkeypatch):
        claim = tmp_path / "claim.json"
        claim.write_text('{"claim_number": "C1"}')
        monkeypatch.setattr(sys, "argv", ["main.py", str(claim)])
        chroma = tmp_path / "chroma_db"
        chroma.mkdir()
        with (
            patch("main.CHROMA_DIR", str(chroma)),
            patch("main.reset_state"),
            patch("main.build_agent") as mock_build,
        ):
            mock_build.return_value.run = MagicMock()
            from main import main
            main()
        mock_build.return_value.run.assert_called_once()


# ---------------------------------------------------------------------------
# Ingestion gating
# ---------------------------------------------------------------------------

class TestIngestionGating:

    def test_ingest_called_when_chroma_dir_missing(self, tmp_path, monkeypatch):
        claim = tmp_path / "claim.json"
        claim.write_text('{}')
        monkeypatch.setattr(sys, "argv", ["main.py", str(claim)])
        missing_dir = str(tmp_path / "no_chroma")
        with (
            patch("main.CHROMA_DIR", missing_dir),
            patch("main.ingest") as mock_ingest,
            patch("main.reset_state"),
            patch("main.build_agent") as mock_build,
        ):
            mock_build.return_value.run = MagicMock()
            from main import main
            main()
        mock_ingest.assert_called_once()

    def test_ingest_skipped_when_chroma_dir_exists(self, tmp_path, monkeypatch):
        claim = tmp_path / "claim.json"
        claim.write_text('{}')
        monkeypatch.setattr(sys, "argv", ["main.py", str(claim)])
        chroma = tmp_path / "chroma_db"
        chroma.mkdir()
        with (
            patch("main.CHROMA_DIR", str(chroma)),
            patch("main.ingest") as mock_ingest,
            patch("main.reset_state"),
            patch("main.build_agent") as mock_build,
        ):
            mock_build.return_value.run = MagicMock()
            from main import main
            main()
        mock_ingest.assert_not_called()


# ---------------------------------------------------------------------------
# build_agent
# ---------------------------------------------------------------------------

class TestBuildAgent:

    def test_system_prompt_is_set(self):
        with patch("main.OpenAIServerModel", return_value=MagicMock()):
            from main import build_agent
            from prompts import SYSTEM_PROMPT
            agent = build_agent()
        assert agent.prompt_templates["system_prompt"] == SYSTEM_PROMPT

    def test_all_five_tools_registered(self):
        with patch("main.OpenAIServerModel", return_value=MagicMock()):
            from main import build_agent
            agent = build_agent()
        tool_names = set(agent.tools.keys())
        assert {
            "importClaims",
            "validateClaims",
            "getRelevantPolicyInfo",
            "processClaim",
            "finalDecision",
        }.issubset(tool_names)

    def test_model_uses_env_credentials(self):
        with patch("main.OpenAIServerModel") as mock_cls:
            mock_cls.return_value = MagicMock()
            with (
                patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test", "OPENAI_API_BASE": "https://api.test.com"}),
            ):
                from main import build_agent
                build_agent()
        mock_cls.assert_called_once_with(
            model_id="gpt-4o-mini",
            api_key="sk-test",
            api_base="https://api.test.com",
        )

    def test_reset_state_called_before_agent_run(self, tmp_path, monkeypatch):
        claim = tmp_path / "claim.json"
        claim.write_text('{}')
        monkeypatch.setattr(sys, "argv", ["main.py", str(claim)])
        chroma = tmp_path / "chroma"
        chroma.mkdir()
        call_order = []
        with (
            patch("main.CHROMA_DIR", str(chroma)),
            patch("main.reset_state", side_effect=lambda: call_order.append("reset")),
            patch("main.build_agent") as mock_build,
        ):
            mock_agent = MagicMock()
            mock_agent.run.side_effect = lambda *a, **kw: call_order.append("run")
            mock_build.return_value = mock_agent
            from main import main
            main()
        assert call_order.index("reset") < call_order.index("run")
