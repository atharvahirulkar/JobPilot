"""Voice layer for the mock interview: the interviewer speaks, you answer out loud.

    questions generated ──► prefetch() synthesizes all of them in parallel
                              (cache-backed, so the session starts instantly
                               and repeated questions cost nothing)
    each question     ──► speak_question()  → MP3 played in the dashboard
    your spoken answer──► transcribe()      → editable transcript → LLM judge
    each evaluation   ──► speak_feedback()  → short spoken verdict

Every method degrades gracefully: if voice is disabled or ElevenLabs fails,
it returns None and the session continues in text mode.
"""
from __future__ import annotations

import logging
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from .client import ElevenLabsClient, VoiceError
from .config import VoiceConfig
from .telemetry import Sink, db_sink

log = logging.getLogger(__name__)

_MAX_SPOKEN_CHARS = 600

_MIME_BY_EXT = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "m4a": "audio/mp4",
    "webm": "audio/webm",
    "ogg": "audio/ogg",
}


def clean_for_speech(text: str, limit: int = _MAX_SPOKEN_CHARS) -> str:
    """Strip markdown and other things that sound wrong when read aloud."""
    t = str(text or "")
    t = re.sub(r"```.*?```", " ", t, flags=re.S)            # code blocks
    t = re.sub(r"`([^`]*)`", r"\1", t)                       # inline code
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)           # md links
    t = re.sub(r"https?://\S+", " ", t)                      # bare URLs
    t = re.sub(r"[*_#>|]+", " ", t)                          # md emphasis/headers
    t = re.sub(r"^\s*(?:\d+[.)]|[-•])\s+", "", t, flags=re.M)  # list markers
    t = " ".join(t.split())
    if len(t) > limit:
        cut = t[:limit]
        end = max(cut.rfind(". "), cut.rfind("? "))
        t = cut[: end + 1] if end > limit // 2 else cut.rsplit(" ", 1)[0] + "."
    return t


def question_script(question: str, index: int, total: int, job: Optional[Dict[str, Any]] = None) -> str:
    """What the interviewer actually says for a question."""
    q = clean_for_speech(question)
    if index == 0:
        role = (job or {}).get("title") or "this role"
        company = (job or {}).get("company")
        where = f" at {company}" if company else ""
        return f"Hi, I'll be your interviewer today for the {role} position{where}. We'll go through {total} questions. First question. {q}"
    if index == total - 1:
        return f"Last question. {q}"
    return f"Question {index + 1}. {q}"


def feedback_script(result: Dict[str, Any]) -> str:
    """A short spoken verdict: score, one strength, one thing to fix."""
    score = float(result.get("overall_score", 0) or 0)
    parts = [f"That scored {score:.0f} out of 10."]
    strengths = result.get("strengths") or []
    weaknesses = result.get("weaknesses") or []
    if strengths:
        parts.append(f"What worked: {clean_for_speech(strengths[0], 160)}")
    if weaknesses:
        parts.append(f"To improve: {clean_for_speech(weaknesses[0], 160)}")
    return " ".join(p if p.endswith((".", "?", "!")) else p + "." for p in parts)


def mime_for(filename: str, fallback: str = "audio/wav") -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return _MIME_BY_EXT.get(ext, fallback)


class VoiceInterviewer:
    def __init__(
        self,
        config: Optional[VoiceConfig] = None,
        *,
        client: Optional[ElevenLabsClient] = None,
        sink: Optional[Sink] = None,
        db_url: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self.config = config or VoiceConfig.from_env()
        self.session_id = session_id or f"voice_{uuid.uuid4().hex[:10]}"
        self.last_error: Optional[str] = None
        self._client = client
        if self._client is None and self.config.enabled:
            try:
                self._client = ElevenLabsClient(self.config, sink=sink or db_sink(db_url))
            except VoiceError as exc:
                self.last_error = str(exc)

    @property
    def enabled(self) -> bool:
        return self._client is not None

    # -------------------------------------------------------------- speaking

    def speak(self, text: str) -> Optional[bytes]:
        if not self.enabled or not text:
            return None
        try:
            return self._client.synthesize(text, session_id=self.session_id)  # type: ignore[union-attr]
        except VoiceError as exc:
            self.last_error = str(exc)
            log.warning("TTS failed, falling back to text: %s", exc)
            return None

    def speak_question(self, question: str, index: int, total: int, job: Optional[Dict[str, Any]] = None) -> Optional[bytes]:
        return self.speak(question_script(question, index, total, job))

    def speak_feedback(self, result: Dict[str, Any]) -> Optional[bytes]:
        return self.speak(feedback_script(result))

    def prefetch(self, questions: List[str], job: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
        """Synthesize every question concurrently so playback never waits.

        Results land in the audio cache, so later speak_question() calls are
        cache hits. Returns {"ok": n, "failed": m}.
        """
        stats = {"ok": 0, "failed": 0}
        if not self.enabled or not questions:
            return stats
        total = len(questions)
        scripts = [question_script(q, i, total, job) for i, q in enumerate(questions)]
        workers = max(1, min(self.config.prefetch_workers, total))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tts-prefetch") as pool:
            futures = [pool.submit(self.speak, s) for s in scripts]
            for fut in as_completed(futures):
                stats["ok" if fut.result() else "failed"] += 1
        return stats

    # ------------------------------------------------------------- listening

    def transcribe(self, audio: bytes, filename: str = "answer.wav") -> Optional[str]:
        if not self.enabled or not audio:
            return None
        try:
            return self._client.transcribe(  # type: ignore[union-attr]
                audio, filename=filename, mime=mime_for(filename), session_id=self.session_id,
            )
        except VoiceError as exc:
            self.last_error = str(exc)
            log.warning("STT failed, falling back to typed answer: %s", exc)
            return None
