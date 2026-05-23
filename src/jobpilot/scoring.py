from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Dict, Iterable, List

from .embeddings import EmbeddingClient


@dataclass
class ScoreResult:
    score: int
    similarity: float
    overlap: List[str]
    missing_skills: List[str]
    rationale: str


class ScoringEngine:
    """Local-first JD scoring engine.

    The engine combines lexical overlap with embedding similarity so it can
    run offline while still producing a useful rank signal.
    """

    def __init__(self, embedding_client: EmbeddingClient | None = None):
        self.embedding_client = embedding_client or EmbeddingClient()

    def score(self, resume_text: str, jd: Dict[str, Any]) -> ScoreResult:
        resume_tokens = self._normalize_tokens(resume_text)
        jd_text = self._build_jd_text(jd)
        jd_tokens = self._normalize_tokens(jd_text)

        overlap = sorted(set(resume_tokens) & set(jd_tokens))
        missing = sorted(set(jd_tokens) - set(resume_tokens))

        similarity = self._cosine_similarity(
            self.embedding_client.embed_texts([resume_text, jd_text])
        )

        lexical_ratio = len(overlap) / max(1, len(set(jd_tokens)))
        blended = 0.65 * similarity + 0.35 * lexical_ratio
        score = max(0, min(100, int(round(blended * 100))))

        rationale = self._build_rationale(score, overlap, missing, similarity)
        return ScoreResult(
            score=score,
            similarity=round(similarity, 4),
            overlap=overlap[:20],
            missing_skills=missing[:20],
            rationale=rationale,
        )

    def _build_jd_text(self, jd: Dict[str, Any]) -> str:
        parts: List[str] = [str(jd.get("title", "")), str(jd.get("description", ""))]
        parts.extend(jd.get("skills", []) or [])
        parts.extend(jd.get("responsibilities", []) or [])
        return "\n".join(part for part in parts if part)

    def _normalize_tokens(self, text: str) -> List[str]:
        tokens = []
        for raw in text.lower().replace("/", " ").replace("+", " ").split():
            token = "".join(ch for ch in raw if ch.isalnum() or ch in {"-"})
            if len(token) > 2:
                tokens.append(token)
        return tokens

    def _cosine_similarity(self, vectors: Iterable[Iterable[float]]) -> float:
        a, b = [list(v) for v in vectors]
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sqrt(sum(x * x for x in a))
        norm_b = sqrt(sum(y * y for y in b))
        if not norm_a or not norm_b:
            return 0.0
        return dot / (norm_a * norm_b)

    def _build_rationale(self, score: int, overlap: List[str], missing: List[str], similarity: float) -> str:
        overlap_preview = ", ".join(overlap[:5]) if overlap else "few direct keyword overlaps"
        missing_preview = ", ".join(missing[:5]) if missing else "no major gaps extracted"
        return (
            f"Score {score}/100 from embedding similarity {similarity:.2f}. "
            f"Overlap: {overlap_preview}. Missing: {missing_preview}."
        )


if __name__ == "__main__":
    from jobpilot.jd_parser import JDParser

    resume = "Data scientist with Python, SQL, experimentation, and model deployment experience."
    jd_text = "Data Scientist\nSkills: Python, SQL, experimentation, statistics, communication"
    jd = JDParser().parse(jd_text)
    result = ScoringEngine().score(resume, jd)
    print(result)
