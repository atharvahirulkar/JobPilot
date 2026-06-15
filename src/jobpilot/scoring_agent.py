"""W3: LLM alignment agent — resume ↔ JD → match score + structured gaps.

Uses GPT-4o-mini with JSON mode for deterministic structured output.
Falls back to local ScoringEngine (embedding cosine) when OPENAI_API_KEY is absent.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

FAANG = {"meta", "google", "apple", "amazon", "netflix", "microsoft", "deepmind", "openai", "anthropic"}
GROWTH = {
    "stripe", "airbnb", "doordash", "uber", "lyft", "coinbase", "robinhood", "figma",
    "notion", "databricks", "snowflake", "palantir", "pinterest", "reddit", "instacart",
    "rivian", "waymo", "cruise", "scale", "cohere", "hugging face", "huggingface",
}

_SYSTEM_PROMPT = """\
You are a strict resume-to-job-description alignment evaluator.
Given a candidate resume and a job description, return ONLY valid JSON — no markdown, no explanation.

JSON schema:
{
  "match_score": <integer 0-100>,
  "matched_skills": [<string>, ...],
  "skill_gaps": [<string>, ...],
  "role_fit": <"DS" | "MLE" | "DE" | "SWE" | "Quant" | "Analyst" | "Research" | "Other">,
  "company_tier": <"FAANG" | "Growth" | "Startup">
}

Scoring rubric:
- 90-100: Near-perfect fit, candidate has all required skills and relevant experience
- 75-89: Strong fit, 1-2 minor gaps that are learnable
- 50-74: Moderate fit, real skill gaps but solid foundation
- 25-49: Weak fit, significant gaps in core requirements
- 0-24: Mismatch, missing most required skills

role_fit classification:
- DS:       Data Scientist (statistics, experimentation, modeling, insights)
- MLE:      ML Engineer (production ML, MLOps, model serving, pipelines)
- DE:       Data Engineer (pipelines, ETL, Spark, dbt, Airflow, warehousing)
- SWE:      Software Engineer (backend, infra, distributed systems, APIs)
- Quant:    Quantitative Researcher / Analyst (finance, trading, statistics, alpha research)
- Analyst:  Data/Business Analyst (SQL, dashboards, reporting, BI)
- Research: ML Researcher (papers, novel methods, PhD-level)
- Other:    Product, PM, DevOps, or unclear

company_tier classification:
- FAANG: Meta, Google, Apple, Amazon, Netflix, Microsoft, DeepMind, OpenAI, Anthropic
- Growth: Well-known scale-ups (Stripe, Databricks, Airbnb, Snowflake, etc.)
- Startup: Early/mid-stage or unknown companies
"""

_USER_TEMPLATE = """\
RESUME (truncated to 1500 chars):
{resume}

JOB TITLE: {title}
COMPANY: {company}
JOB DESCRIPTION (truncated to 2000 chars):
{jd}
REQUIRED SKILLS: {skills}

Return the JSON now.
"""


@dataclass
class AlignmentResult:
    match_score: int
    matched_skills: List[str]
    skill_gaps: List[str]
    role_fit: str
    company_tier: str
    source: str = "llm"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "match_score": self.match_score,
            "matched_skills": self.matched_skills,
            "skill_gaps": self.skill_gaps,
            "role_fit": self.role_fit,
            "company_tier": self.company_tier,
            "source": self.source,
        }

    @property
    def is_top_match(self) -> bool:
        return self.match_score >= 75


def _classify_company_tier(company: str) -> str:
    name = company.lower().strip()
    if any(f in name for f in FAANG):
        return "FAANG"
    if any(g in name for g in GROWTH):
        return "Growth"
    return "Startup"


def _local_fallback(resume_text: str, job: Dict[str, Any]) -> AlignmentResult:
    """Use embedding-cosine ScoringEngine when OpenAI is unavailable."""
    from .scoring import ScoringEngine

    engine = ScoringEngine()
    result = engine.score(resume_text, job)

    # Convert ScoringEngine output to AlignmentResult format
    matched = [kw for kw in result.overlap if len(kw) > 2][:15]
    gaps = result.missing_skills[:10]
    tier = _classify_company_tier(str(job.get("company", "")))

    # Rough role_fit heuristic from title
    title_lower = str(job.get("title", "")).lower()
    if any(w in title_lower for w in ["quant", "quantitative"]):
        role_fit = "Quant"
    elif any(w in title_lower for w in ["research scientist", "researcher", "research engineer"]):
        role_fit = "Research"
    elif any(w in title_lower for w in ["ml engineer", "machine learning engineer", "mlops", "ai engineer",
                                         "applied scientist", "applied ml"]):
        role_fit = "MLE"
    elif any(w in title_lower for w in ["data engineer", "analytics engineer", "etl", "platform engineer"]):
        role_fit = "DE"
    elif any(w in title_lower for w in ["software engineer", "backend engineer", "swe", "sde",
                                         "software developer"]):
        role_fit = "SWE"
    elif any(w in title_lower for w in ["analyst", "analytics", "bi engineer", "business intelligence"]):
        role_fit = "Analyst"
    elif any(w in title_lower for w in ["data scientist", "scientist"]):
        role_fit = "DS"
    else:
        role_fit = "Other"

    return AlignmentResult(
        match_score=result.score,
        matched_skills=matched,
        skill_gaps=gaps,
        role_fit=role_fit,
        company_tier=tier,
        source="local",
    )


class ScoringAgent:
    """LLM alignment agent: resume ↔ JD → structured match score + gaps.

    [INTERVIEW GOLD] Uses GPT-4o-mini JSON mode (response_format=json_object) to get
    deterministic structured output without brittle regex parsing. Falls back to local
    embedding similarity when the API key is absent, so the pipeline works offline.
    """

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        self._client: Optional[Any] = None

    def _get_client(self) -> Optional[Any]:
        if self._client is not None:
            return self._client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key)
        except ImportError:
            self._client = None
        return self._client

    def score(self, resume_text: str, job: Dict[str, Any]) -> AlignmentResult:
        """Score a resume against a job posting.

        Returns AlignmentResult with match_score, matched_skills, skill_gaps,
        role_fit, company_tier. Falls back to local scoring if OpenAI unavailable.
        """
        client = self._get_client()
        if client is None:
            return _local_fallback(resume_text, job)

        user_msg = _USER_TEMPLATE.format(
            resume=resume_text[:1500],
            title=job.get("title", ""),
            company=job.get("company", ""),
            jd=str(job.get("description", ""))[:2000],
            skills=", ".join(str(s) for s in (job.get("skills") or [])),
        )

        try:
            response = client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            )
            raw = response.choices[0].message.content or "{}"
            data = json.loads(raw)
        except Exception:
            return _local_fallback(resume_text, job)

        return AlignmentResult(
            match_score=max(0, min(100, int(data.get("match_score", 0)))),
            matched_skills=list(data.get("matched_skills") or []),
            skill_gaps=list(data.get("skill_gaps") or []),
            role_fit=str(data.get("role_fit") or "Other"),
            company_tier=str(data.get("company_tier") or _classify_company_tier(str(job.get("company", "")))),
            source="llm",
        )

    def score_batch(self, resume_text: str, jobs: List[Dict[str, Any]]) -> List[AlignmentResult]:
        """Score a list of jobs against a resume. Returns results in same order."""
        return [self.score(resume_text, job) for job in jobs]


if __name__ == "__main__":
    import sys
    from pathlib import Path

    resume = Path(sys.argv[1]).read_text() if len(sys.argv) > 1 else "Data Scientist with Python and SQL experience."
    job = {
        "title": "Senior Data Scientist",
        "company": "Stripe",
        "description": "We need Python, SQL, experimentation, causal inference, and statistics.",
        "skills": ["Python", "SQL", "experimentation", "statistics"],
    }
    agent = ScoringAgent()
    result = agent.score(resume, job)
    import pprint
    pprint.pprint(result.to_dict())
    print(f"\nis_top_match: {result.is_top_match}")
