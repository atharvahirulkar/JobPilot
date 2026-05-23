"""W5 tests: candidate model, interview agent nodes, LangGraph routing."""
from __future__ import annotations

import json
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from jobpilot.agents.interview_agent import (
    _fallback_evaluation,
    _fallback_questions,
    evaluate_answer,
    generate_questions,
)
from jobpilot.candidate_model.model import CandidateModelManager
from jobpilot.candidate_model.updater import CandidateModelUpdater, _score_to_delta
from jobpilot.pipeline.graph import (
    JobPilotState,
    _should_continue,
    build_graph,
    evaluator_node,
    model_updater_node,
    question_generator_node,
)

SQLITE_URL = "sqlite://"  # in-memory; re-created per test via fixture

SAMPLE_JOB: Dict[str, Any] = {
    "title": "ML Engineer",
    "company": "Acme AI",
    "description": "Build ML pipelines. Deploy models. Python, PyTorch, Kubernetes.",
    "skills": ["python", "pytorch", "kubernetes", "mlflow"],
}

WEAK_PROFILE = {"python": 0.3, "pytorch": 0.4, "kubernetes": 0.2}
STRONG_PROFILE = {"python": 0.9, "pytorch": 0.85}


# ---------------------------------------------------------------------------
# CandidateModelManager
# ---------------------------------------------------------------------------

@pytest.fixture
def mgr():
    return CandidateModelManager(db_url=SQLITE_URL)


def test_candidate_model_upsert_new_skill(mgr):
    mgr.upsert_skill("python", confidence_score=0.7)
    profile = mgr.get_profile()
    assert "python" in profile
    assert abs(profile["python"] - 0.7) < 0.01


def test_candidate_model_update_delta(mgr):
    mgr.upsert_skill("pytorch", confidence_score=0.5)
    mgr.upsert_skill("pytorch", confidence_score=0.5, delta=0.1)
    profile = mgr.get_profile()
    assert abs(profile["pytorch"] - 0.6) < 0.01


def test_candidate_model_clamps_at_1(mgr):
    mgr.upsert_skill("sql", confidence_score=0.95, delta=0.2)
    profile = mgr.get_profile()
    assert profile["sql"] <= 1.0


def test_candidate_model_clamps_at_01(mgr):
    mgr.upsert_skill("sql", confidence_score=0.05, delta=-0.5)
    profile = mgr.get_profile()
    assert profile["sql"] >= 0.1


def test_get_weak_skills_threshold(mgr):
    mgr.upsert_skill("strong_skill", confidence_score=0.9)
    mgr.upsert_skill("weak_skill",   confidence_score=0.3)
    weak = mgr.get_weak_skills()
    names = [s["skill_name"] for s in weak]
    assert "weak_skill" in names
    assert "strong_skill" not in names


def test_seed_from_job_creates_skills(mgr):
    mgr.seed_from_job(SAMPLE_JOB)
    profile = mgr.get_profile()
    assert "python" in profile
    assert "pytorch" in profile


def test_seed_from_job_does_not_overwrite(mgr):
    mgr.upsert_skill("python", confidence_score=0.9)
    mgr.seed_from_job(SAMPLE_JOB)
    profile = mgr.get_profile()
    assert abs(profile["python"] - 0.9) < 0.01


# ---------------------------------------------------------------------------
# CandidateModelUpdater
# ---------------------------------------------------------------------------

def test_score_to_delta_strong():
    assert _score_to_delta(9.0) == +0.10


def test_score_to_delta_adequate():
    assert _score_to_delta(7.0) == +0.05


def test_score_to_delta_neutral():
    assert _score_to_delta(5.0) == 0.00


def test_score_to_delta_weak():
    assert _score_to_delta(2.0) == -0.04


def test_updater_updates_skills():
    updater = CandidateModelUpdater(db_url=SQLITE_URL)
    results = [
        {"overall_score": 8.0, "skill_tags": ["python", "mlflow"]},
        {"overall_score": 4.0, "skill_tags": ["kubernetes"]},
    ]
    summary = updater.update_from_session(results, SAMPLE_JOB)
    assert summary["total_questions"] == 2
    assert "python" in summary["updated"]
    assert "mlflow" in summary["updated"]


def test_updater_empty_results():
    updater = CandidateModelUpdater(db_url=SQLITE_URL)
    summary = updater.update_from_session([], SAMPLE_JOB)
    assert summary["updated"] == []
    assert summary["session_score"] == 0.0


# ---------------------------------------------------------------------------
# generate_questions
# ---------------------------------------------------------------------------

def test_generate_questions_fallback_no_key():
    qs = generate_questions(SAMPLE_JOB, WEAK_PROFILE, n_questions=4, api_key="")
    assert isinstance(qs, list)
    assert len(qs) >= 1
    assert all(isinstance(q, str) and len(q) > 5 for q in qs)


def test_generate_questions_with_llm():
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = json.dumps({
        "questions": ["Q1?", "Q2?", "Q3?", "Q4?"]
    })
    with patch("jobpilot.agents.interview_agent.OpenAI") as MockOAI:
        MockOAI.return_value.chat.completions.create.return_value = mock_resp
        qs = generate_questions(SAMPLE_JOB, WEAK_PROFILE, n_questions=4, api_key="sk-fake")
    assert qs == ["Q1?", "Q2?", "Q3?", "Q4?"]


def test_generate_questions_llm_exception_falls_back():
    with patch("jobpilot.agents.interview_agent.OpenAI") as MockOAI:
        MockOAI.return_value.chat.completions.create.side_effect = RuntimeError("timeout")
        qs = generate_questions(SAMPLE_JOB, WEAK_PROFILE, n_questions=4, api_key="sk-fake")
    assert isinstance(qs, list)
    assert len(qs) >= 1


# ---------------------------------------------------------------------------
# evaluate_answer
# ---------------------------------------------------------------------------

def test_evaluate_answer_fallback_no_key():
    result = evaluate_answer("What is XGBoost?", "A gradient boosted tree ensemble.", SAMPLE_JOB, api_key="")
    assert "relevance" in result
    assert "depth" in result
    assert "accuracy" in result
    assert 0 < result["overall_score"] <= 10


def test_evaluate_answer_with_llm():
    llm_output = {
        "relevance": 8, "depth": 7, "accuracy": 9,
        "overall_score": 8.0,
        "strengths": ["clear explanation"],
        "weaknesses": ["missing math"],
        "improved_answer": "Add the loss function derivation.",
    }
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = json.dumps(llm_output)
    with patch("jobpilot.agents.interview_agent.OpenAI") as MockOAI:
        MockOAI.return_value.chat.completions.create.return_value = mock_resp
        result = evaluate_answer("What is XGBoost?", "An ensemble method.", SAMPLE_JOB,
                                  api_key="sk-fake", db_url=SQLITE_URL)
    assert result["relevance"] == 8
    assert result["overall_score"] == 8.0


def test_evaluate_answer_clamps_scores():
    llm_output = {"relevance": 15, "depth": 0, "accuracy": 11,
                  "overall_score": 8.0, "strengths": [], "weaknesses": [], "improved_answer": ""}
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = json.dumps(llm_output)
    with patch("jobpilot.agents.interview_agent.OpenAI") as MockOAI:
        MockOAI.return_value.chat.completions.create.return_value = mock_resp
        result = evaluate_answer("Q", "A", SAMPLE_JOB, api_key="sk-fake", db_url=SQLITE_URL)
    assert result["relevance"] <= 10
    assert result["depth"] >= 1
    assert result["accuracy"] <= 10


# ---------------------------------------------------------------------------
# LangGraph routing + node unit tests
# ---------------------------------------------------------------------------

def _base_state(**kwargs) -> JobPilotState:
    state: JobPilotState = {
        "current_job":         SAMPLE_JOB,
        "generated_questions": ["Q1?", "Q2?", "Q3?"],
        "current_q_index":     0,
        "user_answers":        [],
        "evaluation_results":  [],
        "candidate_profile":   WEAK_PROFILE,
        "session_summary":     {},
        "n_questions":         3,
        "db_url":              SQLITE_URL,
        "api_key":             "",
    }
    state.update(kwargs)
    return state


def test_should_continue_when_questions_remain():
    state = _base_state(current_q_index=1, generated_questions=["Q1?", "Q2?", "Q3?"])
    assert _should_continue(state) == "continue"


def test_should_continue_done_when_all_answered():
    state = _base_state(current_q_index=3, generated_questions=["Q1?", "Q2?", "Q3?"])
    assert _should_continue(state) == "done"


def test_evaluator_node_increments_index():
    state = _base_state(
        current_q_index=0,
        user_answers=["My answer to Q1"],
    )
    result = evaluator_node(state)
    assert result["current_q_index"] == 1
    assert len(result["evaluation_results"]) == 1


def test_evaluator_node_appends_result():
    state = _base_state(
        current_q_index=1,
        generated_questions=["Q1?", "Q2?"],
        user_answers=["A1", "A2"],
        evaluation_results=[{"overall_score": 7.0, "skill_tags": []}],
    )
    result = evaluator_node(state)
    assert result["current_q_index"] == 2
    assert len(result["evaluation_results"]) == 2


def test_model_updater_node_returns_summary():
    state = _base_state(
        evaluation_results=[
            {"overall_score": 8.0, "skill_tags": ["python"]},
            {"overall_score": 6.0, "skill_tags": ["pytorch"]},
        ],
    )
    result = model_updater_node(state)
    assert "session_summary" in result
    assert result["session_summary"]["total_questions"] == 2


def test_build_graph_returns_compiled_graph():
    graph = build_graph()
    assert graph is not None
    # Graph should have nodes registered
    assert "question_generator" in graph.nodes
    assert "answer_collector"   in graph.nodes
    assert "evaluator"          in graph.nodes
    assert "model_updater"      in graph.nodes
