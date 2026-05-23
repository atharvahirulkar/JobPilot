from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from jobpilot.db_models import CandidateModel, init_db


class CandidateModelManager:
    """Read/write skill confidence scores in the candidate_model table.

    Confidence scores range 0.0–1.0:
      < 0.4  → weak (question generator prioritises these)
      0.4–0.7 → developing
      > 0.7  → strong
    """

    WEAK_THRESHOLD = 0.6

    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or os.getenv(
            "DATABASE_URL", "sqlite:///jobpilot_w5.db"
        )
        self.SessionLocal = init_db(self.db_url)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_all_skills(self) -> List[Dict[str, Any]]:
        with self.SessionLocal() as session:
            rows = session.query(CandidateModel).order_by(
                CandidateModel.confidence_score.asc()
            ).all()
            return [_row_to_dict(r) for r in rows]

    def get_weak_skills(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Return up to `limit` skills with confidence below WEAK_THRESHOLD."""
        with self.SessionLocal() as session:
            rows = (
                session.query(CandidateModel)
                .filter(CandidateModel.confidence_score < self.WEAK_THRESHOLD)
                .order_by(CandidateModel.confidence_score.asc())
                .limit(limit)
                .all()
            )
            return [_row_to_dict(r) for r in rows]

    def get_profile(self) -> Dict[str, float]:
        """Return {skill_name: confidence_score} mapping for all tracked skills."""
        return {r["skill_name"]: r["confidence_score"] for r in self.get_all_skills()}

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def upsert_skill(
        self,
        skill_name: str,
        confidence_score: float,
        delta: float = 0.0,
    ) -> None:
        """Insert or update a skill.  If delta != 0, add delta to current score."""
        confidence_score = max(0.1, min(1.0, confidence_score))
        with self.SessionLocal() as session:
            row = session.query(CandidateModel).filter_by(
                skill_name=skill_name
            ).first()
            if row:
                new_score = max(0.1, min(1.0, row.confidence_score + delta)) if delta else confidence_score
                row.confidence_score = new_score
                row.mention_count += 1
                row.last_seen = datetime.utcnow()
            else:
                row = CandidateModel(
                    skill_name=skill_name,
                    confidence_score=confidence_score,
                    mention_count=1,
                    last_seen=datetime.utcnow(),
                )
                session.add(row)
            session.commit()

    def seed_from_job(self, job: Dict[str, Any]) -> None:
        """Seed candidate model with skills from a job posting at confidence 0.5."""
        for skill in job.get("skills", []) or []:
            name = str(skill).strip().lower()
            if not name:
                continue
            with self.SessionLocal() as session:
                exists = session.query(CandidateModel).filter_by(skill_name=name).first()
            if not exists:
                self.upsert_skill(name, confidence_score=0.5)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _row_to_dict(row: CandidateModel) -> Dict[str, Any]:
    return {
        "skill_name": row.skill_name,
        "confidence_score": row.confidence_score,
        "mention_count": row.mention_count,
        "last_seen": str(row.last_seen),
    }
