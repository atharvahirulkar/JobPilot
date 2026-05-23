from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Sequence
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

from .embeddings import EmbeddingClient
from .job_normalize import normalize_job
from .qdrant_ingest import ensure_text_collection
from .scoring import ScoringEngine


@dataclass
class JobRecord:
    job_id: str
    title: str
    company: str
    location: str
    description: str
    source: str = "manual"
    url: str = ""

    def to_payload(self) -> Dict[str, Any]:
        return asdict(self)


class JobRepository:
    """Store and retrieve jobs from Qdrant, then refine with local scoring."""

    def __init__(self, client: QdrantClient, collection_name: str = "jobpilot_jobs", embedding_client: EmbeddingClient | None = None):
        self.client = client
        self.collection_name = collection_name
        self.embedding_client = embedding_client or EmbeddingClient()
        ensure_text_collection(self.client, self.collection_name, self.embedding_client.vector_size)
        self.scoring_engine = ScoringEngine(self.embedding_client)

    def upsert_jobs(self, jobs: Sequence[Dict[str, Any] | JobRecord]) -> List[str]:
        records = [self._coerce_job(job) for job in jobs]
        texts = [self._job_to_text(record) for record in records]
        vectors = self.embedding_client.embed_texts(texts)
        points = []
        for record, vector in zip(records, vectors):
            points.append(PointStruct(id=record.job_id, vector=vector, payload=record.to_payload()))
        self.client.upsert(collection_name=self.collection_name, points=points)
        return [record.job_id for record in records]

    def rank_jobs_for_resume(self, resume_text: str, limit: int = 5) -> List[Dict[str, Any]]:
        query_vector = self.embedding_client.embed_texts([resume_text])[0]
        try:
            candidates = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=max(limit * 3, limit),
                with_payload=True,
            )
        except Exception:
            candidates = []

        ranked = []
        for candidate in candidates:
            payload = candidate.payload or {}
            jd = {
                "title": payload.get("title", ""),
                "description": payload.get("description", ""),
                "skills": payload.get("skills", []),
                "responsibilities": payload.get("responsibilities", []),
            }
            score = self.scoring_engine.score(resume_text, jd)
            ranked.append(
                {
                    "job_id": payload.get("job_id", str(candidate.id)),
                    "title": payload.get("title", ""),
                    "company": payload.get("company", ""),
                    "location": payload.get("location", ""),
                    "source": payload.get("source", ""),
                    "url": payload.get("url", ""),
                    "similarity": round(float(candidate.score or 0.0), 4),
                    "match_score": score.score,
                    "overlap": score.overlap,
                    "missing_skills": score.missing_skills,
                    "rationale": score.rationale,
                }
            )

        ranked.sort(key=lambda item: (item["match_score"], item["similarity"]), reverse=True)
        return ranked[:limit]

    def _coerce_job(self, job: Dict[str, Any] | JobRecord) -> JobRecord:
        if isinstance(job, JobRecord):
            return job
        data = normalize_job(dict(job)).to_dict()
        return JobRecord(
            job_id=str(data.get("job_id") or data.get("id") or uuid4()),
            title=str(data.get("title", "")),
            company=str(data.get("company", "")),
            location=str(data.get("location", "")),
            description=str(data.get("description", "")),
            source=str(data.get("source", "manual")),
            url=str(data.get("url", "")),
        )

    def _job_to_text(self, job: JobRecord) -> str:
        return "\n".join(part for part in [job.title, job.company, job.location, job.description] if part)


if __name__ == "__main__":
    from jobpilot.qdrant_setup import init_qdrant

    client = init_qdrant(collection_name="jobpilot_jobs", vector_size=384)
    repo = JobRepository(client)
    repo.upsert_jobs(
        [
            {
                "title": "Data Scientist",
                "company": "Acme",
                "location": "Remote",
                "description": "Python, SQL, experimentation, dashboards, stakeholder communication",
            },
            {
                "title": "ML Engineer",
                "company": "Beta",
                "location": "New York",
                "description": "Python, model deployment, Docker, AWS, feature stores",
            },
        ]
    )
    print(repo.rank_jobs_for_resume("Python SQL experimentation and model deployment experience"))
