from __future__ import annotations

from typing import Any, Dict, List

from .model import CandidateModelManager

# Score → confidence delta mapping
# [INTERVIEW GOLD] Asymmetric updates: weak answers hurt less than strong ones help.
# Prevents a single bad session from collapsing hard-earned confidence.
_DELTA_TABLE = [
    (8.0, +0.10),   # strong performance
    (6.0, +0.05),   # adequate
    (4.0,  0.00),   # neutral
    (0.0, -0.04),   # needs work
]


def _score_to_delta(overall_score: float) -> float:
    for threshold, delta in _DELTA_TABLE:
        if overall_score >= threshold:
            return delta
    return -0.04


class CandidateModelUpdater:
    """Update skill confidence scores after an interview session.

    For each evaluated answer:
      1. Extract the skills it tested (from question tags + job gaps)
      2. Compute delta from overall score
      3. Upsert into candidate_model table
    """

    def __init__(self, db_url: str | None = None):
        self.model = CandidateModelManager(db_url=db_url)

    def update_from_session(
        self,
        evaluation_results: List[Dict[str, Any]],
        job: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply confidence deltas from a completed session. Returns a summary."""
        if not evaluation_results:
            return {"updated": [], "session_score": 0.0}

        total_score = sum(r.get("overall_score", 0) for r in evaluation_results)
        avg_score   = total_score / len(evaluation_results)

        updated_skills: List[str] = []
        for result in evaluation_results:
            delta = _score_to_delta(result.get("overall_score", 0))
            for skill in result.get("skill_tags", []):
                name = str(skill).strip().lower()
                if not name:
                    continue
                self.model.upsert_skill(name, confidence_score=0.5, delta=delta)
                updated_skills.append(name)

        return {
            "session_score": round(avg_score, 2),
            "updated": sorted(set(updated_skills)),
            "total_questions": len(evaluation_results),
        }
