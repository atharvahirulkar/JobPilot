from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from .db_models import CandidateModel, InterviewAnswer, JobListing, JobScore, init_db


class JobRepository:
    """Manages job and score persistence in PostgreSQL."""

    def __init__(self, db_url: str | None = None):
        self.db_url = db_url or os.getenv("DATABASE_URL", "postgresql://jobpilot:jobpilot@localhost/jobpilot")
        self.SessionLocal = init_db(self.db_url)

    def add_job(self, job_data: Dict[str, Any]) -> int:
        session = self.SessionLocal()
        try:
            job = JobListing(
                job_id=job_data.get("job_id", ""),
                title=job_data.get("title", ""),
                company=job_data.get("company", ""),
                location=job_data.get("location", ""),
                description=job_data.get("description", ""),
                source=job_data.get("source", "manual"),
                url=job_data.get("url", ""),
                skills=str(job_data.get("skills", [])),
                responsibilities=str(job_data.get("responsibilities", [])),
            )
            session.add(job)
            session.commit()
            return job.id
        finally:
            session.close()

    def add_score(self, score_data: Dict[str, Any]) -> int:
        session = self.SessionLocal()
        try:
            score = JobScore(
                job_id=score_data.get("job_id", ""),
                match_score=score_data.get("match_score", 0),
                similarity=score_data.get("similarity", 0.0),
                skill_gaps=str(score_data.get("skill_gaps", [])),
                role_fit=score_data.get("role_fit", ""),
                company_tier=score_data.get("company_tier", ""),
            )
            session.add(score)
            session.commit()
            return score.id
        finally:
            session.close()

    def get_top_jobs(self, limit: int = 10) -> List[Dict[str, Any]]:
        session = self.SessionLocal()
        try:
            jobs = (
                session.query(JobListing)
                .outerjoin(JobScore, JobListing.job_id == JobScore.job_id)
                .order_by(JobScore.match_score.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "id": job.id,
                    "job_id": job.job_id,
                    "title": job.title,
                    "company": job.company,
                    "location": job.location,
                    "source": job.source,
                }
                for job in jobs
            ]
        finally:
            session.close()

    def get_top_jobs_with_scores(self, limit: int = 10, min_score: int = 0) -> List[Dict[str, Any]]:
        """Return top jobs joined with their score data — used by report_agent."""
        session = self.SessionLocal()
        try:
            rows = (
                session.query(JobListing, JobScore)
                .outerjoin(JobScore, JobListing.job_id == JobScore.job_id)
                .filter(JobScore.match_score >= min_score)
                .order_by(JobScore.match_score.desc())
                .limit(limit)
                .all()
            )
            results = []
            for job, score in rows:
                results.append({
                    "job_id":        job.job_id,
                    "title":         job.title,
                    "company":       job.company,
                    "location":      job.location,
                    "source":        job.source,
                    "url":           job.url,
                    "match_score":   score.match_score if score else 0,
                    "skill_gaps":    score.skill_gaps  if score else "[]",
                    "role_fit":      score.role_fit    if score else "",
                    "company_tier":  score.company_tier if score else "",
                })
            return results
        finally:
            session.close()

    def count_jobs_since(self, hours: int = 24) -> int:
        """Count jobs ingested in the last N hours."""
        from datetime import datetime, timedelta
        session = self.SessionLocal()
        try:
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            return session.query(JobListing).filter(JobListing.created_at >= cutoff).count()
        finally:
            session.close()

    def update_candidate_skill(self, skill_name: str, confidence_delta: float = 0.1) -> None:
        session = self.SessionLocal()
        try:
            skill = session.query(CandidateModel).filter_by(skill_name=skill_name).first()
            if skill:
                skill.confidence_score = min(1.0, skill.confidence_score + confidence_delta)
                skill.mention_count += 1
            else:
                skill = CandidateModel(skill_name=skill_name, confidence_score=0.5, mention_count=1)
                session.add(skill)
            session.commit()
        finally:
            session.close()


if __name__ == "__main__":
    repo = JobRepository()
    print("Job repository initialized")
    print(f"DB URL: {repo.db_url}")

# Backwards-compatible alias used by some modules/tests
PersistenceLayer = JobRepository
