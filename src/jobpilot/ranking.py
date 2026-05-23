"""W3: Ranking + top-match flagging for scored jobs.

Pure Python — no I/O, fully unit-testable.

[INTERVIEW GOLD] Separating ranking from scoring is the key design choice here.
Ranking is a deterministic transform on already-scored data, so it never needs
to call an LLM and can be tested in isolation.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .scoring_agent import AlignmentResult


def rank_jobs(
    jobs: List[Dict[str, Any]],
    scores: List[AlignmentResult],
) -> List[Dict[str, Any]]:
    """Merge job dicts with their AlignmentResult and sort by match_score descending.

    Args:
        jobs:   List of job dicts (from job_normalize or DB).
        scores: Parallel list of AlignmentResult in the same order as jobs.

    Returns:
        List of merged dicts with scoring fields attached, sorted high→low.
    """
    if len(jobs) != len(scores):
        raise ValueError(f"jobs ({len(jobs)}) and scores ({len(scores)}) must be same length")

    merged = []
    for job, score in zip(jobs, scores):
        entry = dict(job)
        entry.update(score.to_dict())
        entry["is_top_match"] = score.is_top_match
        merged.append(entry)

    merged.sort(key=lambda x: x["match_score"], reverse=True)
    return merged


def get_top_matches(ranked_jobs: List[Dict[str, Any]], threshold: int = 75) -> List[Dict[str, Any]]:
    """Filter ranked jobs to those at or above the score threshold."""
    return [j for j in ranked_jobs if j.get("match_score", 0) >= threshold]


def summarize_gaps(ranked_jobs: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count how often each skill gap appears across all scored jobs.

    Useful for morning report: "distributed systems appeared as a gap in 8/12 jobs".
    """
    gap_counts: Dict[str, int] = {}
    for job in ranked_jobs:
        for gap in job.get("skill_gaps") or []:
            gap_counts[gap] = gap_counts.get(gap, 0) + 1
    return dict(sorted(gap_counts.items(), key=lambda kv: kv[1], reverse=True))


def print_ranked_summary(ranked_jobs: List[Dict[str, Any]], limit: int = 10) -> None:
    """Print a human-readable ranked summary to stdout."""
    top = get_top_matches(ranked_jobs)
    print(f"\nFound {len(ranked_jobs)} scored jobs | {len(top)} top matches (≥75)\n")
    print(f"{'#':<3} {'Score':>5}  {'Top':>4}  {'Title':<35} {'Company':<25} {'Tier':<8} {'Fit'}")
    print("-" * 95)
    for i, job in enumerate(ranked_jobs[:limit], 1):
        flag = "★" if job.get("is_top_match") else " "
        print(
            f"{i:<3} {job.get('match_score', 0):>5}  {flag:>4}  "
            f"{str(job.get('title', ''))[:34]:<35} "
            f"{str(job.get('company', ''))[:24]:<25} "
            f"{str(job.get('company_tier', '')):<8} "
            f"{job.get('role_fit', '')}"
        )

    gaps = summarize_gaps(ranked_jobs)
    if gaps:
        print("\nRecurring skill gaps:")
        for gap, count in list(gaps.items())[:8]:
            print(f"  {gap} ({count}x)")
