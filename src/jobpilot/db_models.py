from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class JobListing(Base):
    __tablename__ = "job_listings"

    id = Column(Integer, primary_key=True)
    job_id = Column(String(255), unique=True, index=True, nullable=False)
    title = Column(String(255), nullable=False)
    company = Column(String(255), nullable=False)
    location = Column(String(255), nullable=True)
    description = Column(Text, nullable=False)
    source = Column(String(64), nullable=False, default="manual")
    url = Column(String(2048), nullable=True)
    skills = Column(Text, nullable=True)
    responsibilities = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class JobScore(Base):
    __tablename__ = "job_scores"

    id = Column(Integer, primary_key=True)
    job_id = Column(String(255), index=True, nullable=False)
    match_score = Column(Integer, nullable=False)
    similarity = Column(Float, nullable=False)
    skill_gaps = Column(Text, nullable=True)
    role_fit = Column(String(64), nullable=True)
    company_tier = Column(String(64), nullable=True)
    scored_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    resume_version = Column(String(64), nullable=True, default="latest")


class CandidateModel(Base):
    __tablename__ = "candidate_model"

    id = Column(Integer, primary_key=True)
    skill_name = Column(String(255), index=True, nullable=False)
    confidence_score = Column(Float, nullable=False, default=0.5)
    mention_count = Column(Integer, nullable=False, default=0)
    last_seen = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class InterviewAnswer(Base):
    __tablename__ = "interview_answers"

    id = Column(Integer, primary_key=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    score = Column(Integer, nullable=True)
    rationale = Column(Text, nullable=True)
    answered_at = Column(DateTime, nullable=False, default=datetime.utcnow)



class VoiceEvent(Base):
    """One ElevenLabs call (or cache hit): used for latency and cost telemetry."""

    __tablename__ = "voice_events"
    id = Column(Integer, primary_key=True)
    session_id = Column(String(64), index=True, nullable=True)
    op = Column(String(16), nullable=False)          # "tts" | "stt"
    model = Column(String(64), nullable=False)
    chars = Column(Integer, nullable=False, default=0)  # TTS billing unit
    audio_bytes = Column(Integer, nullable=False, default=0)
    ttfb_ms = Column(Float, nullable=True)             # time to first audio byte
    total_ms = Column(Float, nullable=False, default=0.0)
    cache_hit = Column(Integer, nullable=False, default=0)
    status = Column(String(32), nullable=False, default="ok")
    attempts = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

def init_db(db_url: str) -> sessionmaker:
    engine = create_engine(db_url, echo=False)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


if __name__ == "__main__":
    import os

    db_url = os.getenv("DATABASE_URL", "postgresql://jobpilot:jobpilot@localhost/jobpilot")
    SessionLocal = init_db(db_url)
    print(f"Database initialized: {db_url}")
