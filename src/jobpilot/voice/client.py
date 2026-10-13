"""Thin ElevenLabs HTTP client: streaming TTS + Scribe speech-to-text.

Design choices:
  * Streaming endpoint, so time-to-first-audio is measured and minimized.
  * Retries only on 429 / 5xx / network errors, with exponential backoff and
    jitter, honoring Retry-After. 4xx errors (bad key, bad voice) fail fast.
  * Every call emits a VoiceMetric to a pluggable sink (DB by default).
"""
from __future__ import annotations

import logging
import random
import time
from typing import Callable, Optional

import httpx

from .cache import AudioCache
from .config import VoiceConfig
from .telemetry import Sink, VoiceMetric

log = logging.getLogger(__name__)

_RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504}


class VoiceError(RuntimeError):
    def __init__(self, message: str, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.status = status


class ElevenLabsClient:
    def __init__(
        self,
        config: VoiceConfig,
        *,
        cache: Optional[AudioCache] = None,
        sink: Optional[Sink] = None,
        transport: Optional[httpx.BaseTransport] = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not config.enabled:
            raise VoiceError("ELEVENLABS_API_KEY is not set")
        self.config = config
        self.cache = cache if cache is not None else AudioCache(config.cache_dir, config.cache_max_mb)
        self.sink = sink or (lambda _m: None)
        self._sleep = sleep
        self._http = httpx.Client(
            base_url=config.base_url,
            headers={"xi-api-key": config.api_key or ""},
            timeout=httpx.Timeout(config.timeout_s, connect=5.0),
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    # ------------------------------------------------------------------ TTS

    def synthesize(self, text: str, *, session_id: Optional[str] = None) -> bytes:
        """Return MP3 bytes for `text`, from cache when possible."""
        text = " ".join((text or "").split())
        if not text:
            return b""
        cfg = self.config
        settings = cfg.settings.as_payload()
        key = AudioCache.key(
            text, model=cfg.tts_model, voice_id=cfg.voice_id,
            output_format=cfg.output_format, settings=settings,
        )

        t0 = time.perf_counter()
        cached = self.cache.get(key)
        if cached:
            self._emit(VoiceMetric(
                op="tts", model=cfg.tts_model, chars=len(text), audio_bytes=len(cached),
                ttfb_ms=0.0, total_ms=_ms(t0), cache_hit=True, session_id=session_id,
            ))
            return cached

        body = {"text": text, "model_id": cfg.tts_model, "voice_settings": settings}
        url = f"/v1/text-to-speech/{cfg.voice_id}/stream"
        params = {"output_format": cfg.output_format}

        def _call() -> tuple[bytes, float]:
            start = time.perf_counter()
            ttfb: Optional[float] = None
            chunks: list[bytes] = []
            with self._http.stream("POST", url, params=params, json=body) as resp:
                if resp.status_code >= 400:
                    resp.read()
                    raise _http_error(resp)
                for chunk in resp.iter_bytes():
                    if chunk:
                        if ttfb is None:
                            ttfb = _ms(start)
                        chunks.append(chunk)
            return b"".join(chunks), (ttfb if ttfb is not None else _ms(start))

        try:
            (audio, ttfb), attempts = self._with_retries(_call)
        except VoiceError as exc:
            self._emit(VoiceMetric(
                op="tts", model=cfg.tts_model, chars=len(text), total_ms=_ms(t0),
                status=f"error:{exc.status or 'net'}", session_id=session_id,
            ))
            raise

        if not audio:
            raise VoiceError("ElevenLabs returned empty audio")
        self.cache.put(key, audio)
        self._emit(VoiceMetric(
            op="tts", model=cfg.tts_model, chars=len(text), audio_bytes=len(audio),
            ttfb_ms=ttfb, total_ms=_ms(t0), attempts=attempts, session_id=session_id,
        ))
        return audio

    # ------------------------------------------------------------------ STT

    def transcribe(
        self,
        audio: bytes,
        *,
        filename: str = "answer.wav",
        mime: str = "audio/wav",
        session_id: Optional[str] = None,
    ) -> str:
        """Transcribe a spoken answer with ElevenLabs Scribe."""
        if not audio:
            return ""
        cfg = self.config
        t0 = time.perf_counter()

        def _call() -> str:
            resp = self._http.post(
                "/v1/speech-to-text",
                data={"model_id": cfg.stt_model, "tag_audio_events": "false"},
                files={"file": (filename, audio, mime)},
            )
            if resp.status_code >= 400:
                raise _http_error(resp)
            return str(resp.json().get("text", "")).strip()

        try:
            text, attempts = self._with_retries(_call)
        except VoiceError as exc:
            self._emit(VoiceMetric(
                op="stt", model=cfg.stt_model, audio_bytes=len(audio), total_ms=_ms(t0),
                status=f"error:{exc.status or 'net'}", session_id=session_id,
            ))
            raise
        self._emit(VoiceMetric(
            op="stt", model=cfg.stt_model, chars=len(text), audio_bytes=len(audio),
            total_ms=_ms(t0), attempts=attempts, session_id=session_id,
        ))
        return text

    # ------------------------------------------------------------ internals

    def _with_retries(self, fn):
        attempts = 0
        while True:
            attempts += 1
            try:
                return fn(), attempts
            except VoiceError as exc:
                err: VoiceError = exc
                retryable = exc.status in _RETRYABLE
                delay = getattr(exc, "retry_after", None)
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                err = VoiceError(f"network error: {exc}")
                retryable, delay = True, None
            if not retryable or attempts > self.config.max_retries:
                raise err
            backoff = delay if delay is not None else min(8.0, 0.5 * 2 ** (attempts - 1))
            self._sleep(backoff + random.uniform(0, 0.25))

    def _emit(self, metric: VoiceMetric) -> None:
        try:
            self.sink(metric)
        except Exception as exc:  # pragma: no cover
            log.debug("voice sink failed: %s", exc)


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 1)


def _http_error(resp: httpx.Response) -> VoiceError:
    try:
        detail = resp.json().get("detail", resp.text)
    except Exception:
        detail = resp.text
    err = VoiceError(f"ElevenLabs HTTP {resp.status_code}: {str(detail)[:200]}", resp.status_code)
    retry_after = resp.headers.get("retry-after")
    if retry_after:
        try:
            err.retry_after = min(30.0, float(retry_after))  # type: ignore[attr-defined]
        except ValueError:
            pass
    return err
