"""Tests for Chronicle ownership and review governance."""

from __future__ import annotations

from pathlib import Path

import yaml

from chronicle.epoch import SCHEMA_IDS, schema_id


ROOT = Path(__file__).resolve().parents[1]


def test_chronicle_governance_files_define_required_review_surface():
    agents_path = ROOT / ".github" / "chronicle-agents.yml"
    codeowners_path = ROOT / ".github" / "CODEOWNERS"
    template_path = ROOT / ".github" / "pull_request_template.md"
    agents = yaml.safe_load(agents_path.read_text())
    codeowners = codeowners_path.read_text()
    pr_template = template_path.read_text()

    assert agents["schema_version"] in SCHEMA_IDS["approved_agents"].accepted
    assert agents["schema_version"] == schema_id("approved_agents")
    assert agents["owners"]["github_team"] == "PolicyEngine/core-developers"
    assert "@PolicyEngine/core-developers" in codeowners
    assert "/packages/** @PolicyEngine/core-developers" in codeowners
    assert "/policyengine_chronicle/** @PolicyEngine/core-developers" in codeowners

    role_ids = {agent["id"] for agent in agents["approved_agents"]}
    assert {
        "ledger-source-ingestor",
        "ledger-contract-maintainer",
    } <= role_ids
    assert "ledger-target-profile-author" not in role_ids

    required_judges = set(agents["required_judges"])
    assert {
        "ledger-source-fidelity",
        "ledger-contract",
        "ledger-boundary",
    } <= required_judges
    assert "ledger-target-profile" not in required_judges
    for agent in agents["approved_agents"]:
        assert set(agent["required_judges"]) <= required_judges
        assert agent["allowed_paths"]
        assert agent["required_deterministic_checks"]
        assert agent["id"] in pr_template

    assert "Approved Chronicle agent role" in pr_template
    assert "Deterministic checks run" in pr_template
    assert "LLM judge verdicts" in pr_template
    for judge_id in required_judges:
        assert judge_id in pr_template

    boundary_verdict = agents["required_judges"]["ledger-boundary"]["verdict"]
    from chronicle.boundary import MICRODATA_BOUNDARY_CLAUSES

    assert len(MICRODATA_BOUNDARY_CLAUSES) == 3
    assert all(len(clause) > 60 for clause in MICRODATA_BOUNDARY_CLAUSES)
    for clause in MICRODATA_BOUNDARY_CLAUSES:
        assert clause in boundary_verdict
    governance_doc = (ROOT / "docs" / "chronicle-governance.md").read_text()
    for clause in MICRODATA_BOUNDARY_CLAUSES:
        assert clause in governance_doc


def test_ci_runs_boundary_and_governance_tests():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()

    assert "uv run pytest -q" in ci
    assert (ROOT / "tests" / "test_chronicle_boundaries.py").exists()
    assert (ROOT / "tests" / "test_chronicle_consumer_contract.py").exists()
    assert (ROOT / "tests" / "test_chronicle_governance.py").exists()
