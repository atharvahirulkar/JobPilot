"""Unit tests for scoring_agent.py and ranking.py"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from jobpilot.scoring_agent import (
    AlignmentResult,
    ScoringAgent,
    _classify_company_tier,
    _local_fallback,
)
from jobpilot.ranking import get_top_matches, rank_jobs, summarize_gaps


# Stub ScoreResult so tests don't need sentence-transformers installed
class _StubScoreResult:
    def __init__(self):
        self.score = 72
        self.similarity = 0.72
        self.overlap = ["python", "sql", "machine"]
        self.missing_skills = ["causal inference", "statistics"]


# ── Fixtures ──────────────────────────────────────────────────────────────────

RESUME = "Atharva Hirulkar — Data Scientist. Python, SQL, ML, fraud detection, model deployment."

JOB_DS = {
    "job_id": "test_ds_1",
    "title": "Data Scientist",
    "company": "Stripe",
    "location": "Remote",
    "description": "Python, SQL, experimentation, causal inference, statistics, A/B testing",
    "skills": ["Python", "SQL", "statistics"],
}

JOB_MLE = {
    "job_id": "test_mle_1",
    "title": "ML Engineer",
    "company": "Google",
    "location": "Remote",
    "description": "Model serving, MLOps, TensorFlow, Kubernetes, production ML",
    "skills": ["TensorFlow", "Kubernetes", "MLOps"],
}

JOB_STARTUP = {
    "job_id": "test_startup_1",
    "title": "Data Analyst",
    "company": "TinyStartup Inc",
    "location": "NYC",
    "description": "SQL, dashboards, reporting, Tableau",
    "skills": ["SQL", "Tableau"],
}


# ── _classify_company_tier ─────────────────────────────────────────────────────

def test_classify_faang():
    assert _classify_company_tier("Google") == "FAANG"
    assert _classify_company_tier("meta") == "FAANG"
    assert _classify_company_tier("Amazon Web Services") == "FAANG"
    assert _classify_company_tier("Anthropic") == "FAANG"


def test_classify_growth():
    assert _classify_company_tier("Stripe") == "Growth"
    assert _classify_company_tier("Databricks") == "Growth"
    assert _classify_company_tier("Snowflake") == "Growth"


def test_classify_startup():
    assert _classify_company_tier("TinyStartup Inc") == "Startup"
    assert _classify_company_tier("Acme Corp") == "Startup"


# ── AlignmentResult ────────────────────────────────────────────────────────────

def test_alignment_result_is_top_match():
    r = AlignmentResult(75, ["Python"], ["k8s"], "DS", "Growth")
    assert r.is_top_match is True

    r2 = AlignmentResult(74, ["Python"], ["k8s"], "DS", "Growth")
    assert r2.is_top_match is False


def test_alignment_result_to_dict_keys():
    r = AlignmentResult(80, ["Python"], ["k8s"], "MLE", "FAANG", source="llm")
    d = r.to_dict()
    assert set(d) == {"match_score", "matched_skills", "skill_gaps", "role_fit", "company_tier", "source"}
    assert d["match_score"] == 80


# ── local fallback ─────────────────────────────────────────────────────────────

def test_local_fallback_returns_alignment_result():
    with patch("jobpilot.scoring.ScoringEngine") as MockEngine:
        MockEngine.return_value.score.return_value = _StubScoreResult()
        result = _local_fallback(RESUME, JOB_DS)
    assert isinstance(result, AlignmentResult)
    assert 0 <= result.match_score <= 100
    assert result.source == "local"
    assert result.company_tier == "Growth"  # Stripe


def test_local_fallback_role_fit_mle():
    with patch("jobpilot.scoring.ScoringEngine") as MockEngine:
        MockEngine.return_value.score.return_value = _StubScoreResult()
        result = _local_fallback(RESUME, JOB_MLE)
    assert result.role_fit == "MLE"


def test_local_fallback_role_fit_analyst():
    with patch("jobpilot.scoring.ScoringEngine") as MockEngine:
        MockEngine.return_value.score.return_value = _StubScoreResult()
        result = _local_fallback(RESUME, JOB_STARTUP)
    assert result.role_fit == "Analyst"


# ── ScoringAgent — no OpenAI key → uses local fallback ─────────────────────────

def test_scoring_agent_fallback_when_no_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with patch("jobpilot.scoring.ScoringEngine") as MockEngine:
        MockEngine.return_value.score.return_value = _StubScoreResult()
        agent = ScoringAgent()
        result = agent.score(RESUME, JOB_DS)
    assert isinstance(result, AlignmentResult)
    assert result.source == "local"
    assert 0 <= result.match_score <= 100


def test_scoring_agent_batch_length(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with patch("jobpilot.scoring.ScoringEngine") as MockEngine:
        MockEngine.return_value.score.return_value = _StubScoreResult()
        agent = ScoringAgent()
        jobs = [JOB_DS, JOB_MLE, JOB_STARTUP]
        results = agent.score_batch(RESUME, jobs)
    assert len(results) == 3
    assert all(isinstance(r, AlignmentResult) for r in results)


# ── ScoringAgent — mocked OpenAI response ──────────────────────────────────────

def _make_mock_openai(match_score: int = 88):
    """Build a minimal mock that mimics openai.OpenAI().chat.completions.create()."""
    mock_content = json.dumps({
        "match_score": match_score,
        "matched_skills": ["Python", "SQL"],
        "skill_gaps": ["causal inference"],
        "role_fit": "DS",
        "company_tier": "Growth",
    })
    mock_choice = MagicMock()
    mock_choice.message.content = mock_content
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


def test_scoring_agent_uses_llm_when_key_present(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")
    mock_client = _make_mock_openai(match_score=88)

    agent = ScoringAgent()
    agent._client = mock_client

    result = agent.score(RESUME, JOB_DS)
    assert result.match_score == 88
    assert result.source == "llm"
    assert "Python" in result.matched_skills
    assert result.is_top_match is True


def test_scoring_agent_clamps_score(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")
    mock_content = json.dumps({"match_score": 999, "matched_skills": [], "skill_gaps": [], "role_fit": "DS", "company_tier": "FAANG"})
    mock_choice = MagicMock()
    mock_choice.message.content = mock_content
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    agent = ScoringAgent()
    agent._client = mock_client
    result = agent.score(RESUME, JOB_DS)
    assert result.match_score == 100


def test_scoring_agent_falls_back_on_llm_exception(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("network error")

    with patch("jobpilot.scoring.ScoringEngine") as MockEngine:
        MockEngine.return_value.score.return_value = _StubScoreResult()
        agent = ScoringAgent()
        agent._client = mock_client
        result = agent.score(RESUME, JOB_DS)
    assert result.source == "local"


# ── ranking.py ────────────────────────────────────────────────────────────────

def _make_scored_jobs():
    jobs = [JOB_DS.copy(), JOB_MLE.copy(), JOB_STARTUP.copy()]
    scores = [
        AlignmentResult(88, ["Python", "SQL"], ["causal inference"], "DS", "Growth"),
        AlignmentResult(45, ["ML"], ["TensorFlow", "k8s"], "MLE", "FAANG"),
        AlignmentResult(30, ["SQL"], ["Tableau"], "Analyst", "Startup"),
    ]
    return jobs, scores


def test_rank_jobs_sorted_descending():
    jobs, scores = _make_scored_jobs()
    ranked = rank_jobs(jobs, scores)
    assert ranked[0]["match_score"] == 88
    assert ranked[1]["match_score"] == 45
    assert ranked[2]["match_score"] == 30


def test_rank_jobs_attaches_is_top_match():
    jobs, scores = _make_scored_jobs()
    ranked = rank_jobs(jobs, scores)
    assert ranked[0]["is_top_match"] is True
    assert ranked[1]["is_top_match"] is False


def test_rank_jobs_length_mismatch_raises():
    with pytest.raises(ValueError):
        rank_jobs([JOB_DS], [])


def test_get_top_matches_default_threshold():
    jobs, scores = _make_scored_jobs()
    ranked = rank_jobs(jobs, scores)
    top = get_top_matches(ranked)
    assert len(top) == 1
    assert top[0]["match_score"] == 88


def test_get_top_matches_custom_threshold():
    jobs, scores = _make_scored_jobs()
    ranked = rank_jobs(jobs, scores)
    top = get_top_matches(ranked, threshold=40)
    assert len(top) == 2


def test_summarize_gaps():
    jobs, scores = _make_scored_jobs()
    ranked = rank_jobs(jobs, scores)
    gaps = summarize_gaps(ranked)
    assert isinstance(gaps, dict)
    # Each of these gaps appears exactly once
    assert gaps.get("causal inference") == 1
    assert gaps.get("Tableau") == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
