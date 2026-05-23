from __future__ import annotations

"""LangGraph interview loop pipeline.

State machine:
    question_generator → answer_collector → evaluator
                              ↑                  │
                              └── (more Qs) ─────┘
                                                  │
                                         (done) model_updater → END

[INTERVIEW GOLD] Uses LangGraph interrupt() for true human-in-the-loop:
the graph pauses at answer_collector, yields control to the caller,
resumes with the user's answer via Command(resume=...).
This is the canonical agentic pattern — the graph is stateful across turns.
"""

import os
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from jobpilot.agents.interview_agent import evaluate_answer, generate_questions
from jobpilot.candidate_model.model import CandidateModelManager
from jobpilot.candidate_model.updater import CandidateModelUpdater

# Setup LangSmith if configured
_LANGSMITH_ENABLED = bool(os.getenv("LANGSMITH_API_KEY"))
if _LANGSMITH_ENABLED:
    os.environ.setdefault("LANGSMITH_PROJECT", "jobpilot")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class JobPilotState(TypedDict):
    current_job:          Dict[str, Any]
    generated_questions:  List[str]
    current_q_index:      int
    user_answers:         List[str]
    evaluation_results:   List[Dict[str, Any]]
    candidate_profile:    Dict[str, float]
    session_summary:      Dict[str, Any]
    n_questions:          int
    db_url:               Optional[str]
    api_key:              Optional[str]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def question_generator_node(state: JobPilotState) -> Dict[str, Any]:
    """Load candidate model + generate questions weighted toward weak skills."""
    db_url  = state.get("db_url")
    api_key = state.get("api_key") or os.getenv("OPENAI_API_KEY")
    job     = state["current_job"]

    manager = CandidateModelManager(db_url=db_url)
    manager.seed_from_job(job)
    profile = manager.get_profile()

    questions = generate_questions(
        job=job,
        candidate_profile=profile,
        n_questions=state.get("n_questions", 4),
        api_key=api_key,
    )
    return {
        "generated_questions": questions,
        "current_q_index":     0,
        "user_answers":        [],
        "evaluation_results":  [],
        "candidate_profile":   profile,
    }


def answer_collector_node(state: JobPilotState) -> Dict[str, Any]:
    """Pause and wait for the user's answer via interrupt()."""
    idx       = state["current_q_index"]
    questions = state["generated_questions"]
    question  = questions[idx]

    # interrupt() suspends the graph; caller resumes with Command(resume=answer)
    answer = interrupt({
        "question":      question,
        "index":         idx,
        "total":         len(questions),
    })
    return {"user_answers": [*state.get("user_answers", []), str(answer)]}


def evaluator_node(state: JobPilotState) -> Dict[str, Any]:
    """LLM-as-judge: score the latest answer on relevance, depth, accuracy."""
    idx     = state["current_q_index"]
    q       = state["generated_questions"][idx]
    a       = state["user_answers"][idx]
    job     = state["current_job"]
    api_key = state.get("api_key") or os.getenv("OPENAI_API_KEY")
    db_url  = state.get("db_url")

    # Derive skill tags from job skills closest to the question
    skill_tags = [str(s) for s in (job.get("skills") or [])][:3]

    result = evaluate_answer(
        question=q,
        answer=a,
        job=job,
        skill_tags=skill_tags,
        api_key=api_key,
        db_url=db_url,
    )
    return {
        "evaluation_results": [*state.get("evaluation_results", []), result],
        "current_q_index":    idx + 1,
    }


def model_updater_node(state: JobPilotState) -> Dict[str, Any]:
    """Update candidate skill confidence from session results."""
    updater = CandidateModelUpdater(db_url=state.get("db_url"))
    summary = updater.update_from_session(
        evaluation_results=state["evaluation_results"],
        job=state["current_job"],
    )
    return {"session_summary": summary}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def _should_continue(state: JobPilotState) -> str:
    if state["current_q_index"] < len(state["generated_questions"]):
        return "continue"
    return "done"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph(checkpointer=None):
    """Build and compile the interview LangGraph."""
    builder = StateGraph(JobPilotState)

    builder.add_node("question_generator", question_generator_node)
    builder.add_node("answer_collector",   answer_collector_node)
    builder.add_node("evaluator",          evaluator_node)
    builder.add_node("model_updater",      model_updater_node)

    builder.set_entry_point("question_generator")
    builder.add_edge("question_generator", "answer_collector")
    builder.add_edge("answer_collector",   "evaluator")
    builder.add_conditional_edges(
        "evaluator",
        _should_continue,
        {"continue": "answer_collector", "done": "model_updater"},
    )
    builder.add_edge("model_updater", END)

    cp = checkpointer or MemorySaver()
    return builder.compile(checkpointer=cp)


# ---------------------------------------------------------------------------
# CLI runner (used by main.py)
# ---------------------------------------------------------------------------

def run_interview_session(
    job: Dict[str, Any],
    n_questions: int = 4,
    db_url: Optional[str] = None,
    api_key: Optional[str] = None,
    thread_id: str = "default",
) -> Dict[str, Any]:
    """Run a full interview session in the terminal.

    Prints each question, collects answers via stdin, prints scores after each.
    Returns the final session summary.
    """
    graph = build_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: JobPilotState = {
        "current_job":         job,
        "generated_questions": [],
        "current_q_index":     0,
        "user_answers":        [],
        "evaluation_results":  [],
        "candidate_profile":   {},
        "session_summary":     {},
        "n_questions":         n_questions,
        "db_url":              db_url,
        "api_key":             api_key,
    }

    print(f"\n{'─'*60}")
    print(f"  JobPilot Mock Interview")
    print(f"  Role: {job.get('title','?')} @ {job.get('company','?')}")
    print(f"  Questions: {n_questions}")
    print(f"{'─'*60}\n")

    # Kick off the graph — it will run until the first interrupt
    graph.invoke(initial_state, config)

    # Loop: collect answers until graph completes
    while True:
        snap = graph.get_state(config)
        if not snap.next:
            break  # graph finished

        # Unpack interrupt payload
        task = snap.tasks[0] if snap.tasks else None
        interrupt_val = task.interrupts[0].value if (task and task.interrupts) else {}

        q_index = interrupt_val.get("index", 0)
        total   = interrupt_val.get("total", n_questions)
        question = interrupt_val.get("question", "")

        print(f"Q{q_index + 1}/{total}: {question}")
        answer = input("\nYour answer: ").strip()
        if not answer:
            answer = "(no answer provided)"
        print()

        # Resume graph with the answer
        graph.invoke(Command(resume=answer), config)

        # Print evaluation result for last answered question
        current_snap = graph.get_state(config)
        results = current_snap.values.get("evaluation_results", [])
        if results:
            last = results[-1]
            score = last.get("overall_score", 0)
            bar   = "█" * int(score) + "░" * (10 - int(score))
            print(f"  Score: {score:.1f}/10  [{bar}]")
            strengths = last.get("strengths", [])
            weaknesses = last.get("weaknesses", [])
            if strengths:
                print(f"  ✓ {strengths[0]}")
            if weaknesses:
                print(f"  ✗ {weaknesses[0]}")
            improved = last.get("improved_answer", "")
            if improved:
                print(f"  → {improved}")
            print()

    # Final summary
    final = graph.get_state(config).values
    summary = final.get("session_summary", {})

    print(f"\n{'─'*60}")
    print(f"  Session complete!")
    print(f"  Average score : {summary.get('session_score', 0):.1f}/10")
    print(f"  Skills updated: {', '.join(summary.get('updated', [])) or 'none'}")
    print(f"{'─'*60}\n")

    return summary
