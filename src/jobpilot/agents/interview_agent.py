from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional

from openai import OpenAI

from jobpilot.candidate_model.model import CandidateModelManager
from jobpilot.db_models import InterviewAnswer, init_db

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_QUESTION_SYSTEM = """You are a senior technical interviewer for data science and ML engineering roles.
Generate targeted interview questions based on the candidate's weak skill areas and the job requirements.

Question types to mix:
  - Technical: "Explain how X works" / "Walk me through Y"
  - Behavioral (STAR): "Tell me about a time you..."
  - Situational: "How would you approach..."

Output ONLY a JSON object: {"questions": ["q1", "q2", ...]}"""

_EVALUATOR_SYSTEM = """You are an expert ML interview coach evaluating a candidate's answer.

Score each dimension 1–10:
  relevance  — does the answer address the question asked?
  depth      — technical detail and completeness
  accuracy   — factual correctness

Output ONLY a JSON object with these exact keys:
{
  "relevance": <int>,
  "depth": <int>,
  "accuracy": <int>,
  "overall_score": <float>,   // average of the three
  "strengths": ["..."],        // 1-2 things done well
  "weaknesses": ["..."],       // 1-2 gaps
  "improved_answer": "..."     // one concise sentence on what a better answer would add
}"""

# ---------------------------------------------------------------------------
# Question generator
# ---------------------------------------------------------------------------

def generate_questions(
    job: Dict[str, Any],
    candidate_profile: Dict[str, float],
    n_questions: int = 4,
    api_key: Optional[str] = None,
) -> List[str]:
    """Generate interview questions weighted toward weak candidate skills.

    [INTERVIEW GOLD] Weak skills (confidence < 0.6) get 70% of question slots;
    JD-required skills fill the rest. This implements the adaptive difficulty
    loop that makes JobPilot a real candidate model rather than random prep.
    """
    api_key = api_key or os.getenv("OPENAI_API_KEY")

    weak_skills   = [s for s, c in candidate_profile.items() if c < 0.6]
    jd_skills     = [str(s) for s in (job.get("skills") or [])]
    random.shuffle(weak_skills)
    random.shuffle(jd_skills)

    # 70 / 30 split: weak areas vs. JD requirements
    n_weak = max(1, int(n_questions * 0.7))
    focus_skills = (weak_skills[:n_weak] + jd_skills)[:n_questions]

    if not api_key:
        return _fallback_questions(focus_skills, job)

    client = OpenAI(api_key=api_key)
    prompt = (
        f"Role: {job.get('title', 'Data Scientist')} at {job.get('company', 'a tech company')}\n"
        f"JD summary: {str(job.get('description', ''))[:800]}\n"
        f"Candidate weak areas: {', '.join(weak_skills[:5]) or 'unknown'}\n"
        f"Focus skills for this session: {', '.join(focus_skills)}\n\n"
        f"Generate exactly {n_questions} interview questions. Mix technical + behavioral."
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _QUESTION_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
        )
        parsed = json.loads(resp.choices[0].message.content)
        questions = parsed.get("questions", [])
        if questions and len(questions) >= 1:
            return [str(q) for q in questions[:n_questions]]
    except Exception:
        pass

    return _fallback_questions(focus_skills, job)


def _fallback_questions(skills: List[str], job: Dict[str, Any]) -> List[str]:
    title = job.get("title", "data science")
    templates = [
        f"Walk me through how you would approach a {title} problem from scratch.",
        f"Tell me about a project where you used {skills[0] if skills else 'machine learning'} in production.",
        f"How do you evaluate model performance beyond accuracy? Give a concrete example.",
        f"Describe a time you had to debug a complex data pipeline issue under pressure.",
    ]
    return templates


# ---------------------------------------------------------------------------
# LLM-as-judge evaluator
# ---------------------------------------------------------------------------

def evaluate_answer(
    question: str,
    answer: str,
    job: Dict[str, Any],
    skill_tags: Optional[List[str]] = None,
    api_key: Optional[str] = None,
    db_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Score an answer and persist to answer_history.

    Returns the evaluation dict with keys:
      relevance, depth, accuracy, overall_score,
      strengths, weaknesses, improved_answer, skill_tags
    """
    api_key = api_key or os.getenv("OPENAI_API_KEY")

    if not api_key:
        result = _fallback_evaluation(question, answer)
    else:
        result = _llm_evaluate(question, answer, job, api_key)

    result["skill_tags"] = skill_tags or []
    _persist_answer(question, answer, result, db_url)
    return result


def _llm_evaluate(
    question: str,
    answer: str,
    job: Dict[str, Any],
    api_key: str,
) -> Dict[str, Any]:
    client = OpenAI(api_key=api_key)
    prompt = (
        f"Role being interviewed for: {job.get('title', 'Data Scientist')}\n\n"
        f"Question: {question}\n\n"
        f"Candidate answer: {answer}\n\n"
        "Evaluate this answer."
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _EVALUATOR_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        parsed = json.loads(resp.choices[0].message.content)
        # Clamp scores 1–10 and recompute overall
        for key in ("relevance", "depth", "accuracy"):
            parsed[key] = max(1, min(10, int(parsed.get(key, 5))))
        parsed["overall_score"] = round(
            (parsed["relevance"] + parsed["depth"] + parsed["accuracy"]) / 3, 2
        )
        return parsed
    except Exception:
        return _fallback_evaluation(question, answer)


def _fallback_evaluation(question: str, answer: str) -> Dict[str, Any]:
    # Heuristic: longer answers get slightly higher depth score
    depth = min(10, max(1, len(answer.split()) // 15))
    return {
        "relevance": 5,
        "depth": depth,
        "accuracy": 5,
        "overall_score": round((5 + depth + 5) / 3, 2),
        "strengths": ["answer provided"],
        "weaknesses": ["could not evaluate without API key"],
        "improved_answer": "Set OPENAI_API_KEY for detailed feedback.",
    }


def _persist_answer(
    question: str,
    answer: str,
    result: Dict[str, Any],
    db_url: Optional[str],
) -> None:
    url = db_url or os.getenv("DATABASE_URL", "sqlite:///jobpilot_w5.db")
    try:
        SessionLocal = init_db(url)
        with SessionLocal() as session:
            row = InterviewAnswer(
                question=question,
                answer=answer,
                score=int(result.get("overall_score", 0)),
                rationale=json.dumps({
                    "strengths":       result.get("strengths", []),
                    "weaknesses":      result.get("weaknesses", []),
                    "improved_answer": result.get("improved_answer", ""),
                }),
            )
            session.add(row)
            session.commit()
    except Exception:
        pass  # DB persistence is best-effort; don't crash the session
