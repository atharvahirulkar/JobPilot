"""Latency and usage telemetry for voice calls.

Every TTS/STT call (and every cache hit) becomes a VoiceEvent row, so the
dashboard and `jobpilot voice_stats` can report p50/p95 time-to-first-audio,
cache hit rate, and characters billed. Persistence is best-effort: telemetry
must never break an interview.
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Callable, Iterable, List, Optional

log = logging.getLogger(__name__)


@dataclass
class VoiceMetric:
    op: str                       # "tts" | "stt"
    model: str
    chars: int = 0
    audio_bytes: int = 0
    ttfb_ms: Optional[float] = None
    total_ms: float = 0.0
    cache_hit: bool = False
    status: str = "ok"
    attempts: int = 1
    session_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


Sink = Callable[[VoiceMetric], None]


def db_sink(db_url: Optional[str] = None) -> Sink:
    """Return a sink that writes metrics to the JobPilot database."""
    url = db_url or os.getenv("DATABASE_URL", "sqlite:///jobpilot_w5.db")
    session_factory = None

    def _write(metric: VoiceMetric) -> None:
        nonlocal session_factory
        try:
            from jobpilot.db_models import VoiceEvent, init_db

            if session_factory is None:
                session_factory = init_db(url)
            row = asdict(metric)
            row["cache_hit"] = int(metric.cache_hit)
            with session_factory() as session:
                session.add(VoiceEvent(**row))
                session.commit()
        except Exception as exc:  # pragma: no cover - best effort
            log.debug("voice telemetry write failed: %s", exc)

    return _write


def percentile(values: Iterable[float], pct: float) -> Optional[float]:
    data = sorted(v for v in values if v is not None)
    if not data:
        return None
    k = (len(data) - 1) * pct / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return data[int(k)]
    return data[lo] + (data[hi] - data[lo]) * (k - lo)


def summarize(metrics: List[VoiceMetric]) -> dict:
    """Aggregate a list of metrics into the numbers worth looking at."""
    tts = [m for m in metrics if m.op == "tts"]
    stt = [m for m in metrics if m.op == "stt"]
    tts_live = [m for m in tts if not m.cache_hit and m.status == "ok"]
    return {
        "tts_calls": len(tts),
        "tts_cache_hit_rate": round(sum(m.cache_hit for m in tts) / len(tts), 3) if tts else None,
        "tts_ttfb_p50_ms": percentile((m.ttfb_ms for m in tts_live), 50),
        "tts_ttfb_p95_ms": percentile((m.ttfb_ms for m in tts_live), 95),
        "tts_chars_billed": sum(m.chars for m in tts_live),
        "stt_calls": len(stt),
        "stt_p50_ms": percentile((m.total_ms for m in stt if m.status == "ok"), 50),
        "errors": sum(m.status != "ok" for m in metrics),
    }


def load_metrics(db_url: Optional[str] = None, limit: int = 1000) -> List[VoiceMetric]:
    url = db_url or os.getenv("DATABASE_URL", "sqlite:///jobpilot_w5.db")
    from jobpilot.db_models import VoiceEvent, init_db

    with init_db(url)() as session:
        rows = (
            session.query(VoiceEvent)
            .order_by(VoiceEvent.created_at.desc())
            .limit(limit)
            .all()
        )
        return [
            VoiceMetric(
                op=r.op, model=r.model, chars=r.chars, audio_bytes=r.audio_bytes,
                ttfb_ms=r.ttfb_ms, total_ms=r.total_ms, cache_hit=bool(r.cache_hit),
                status=r.status, attempts=r.attempts, session_id=r.session_id,
                created_at=r.created_at,
            )
            for r in rows
        ]
