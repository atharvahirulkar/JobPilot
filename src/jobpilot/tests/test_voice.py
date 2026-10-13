"""W8 tests: ElevenLabs voice layer (TTS, STT, cache, retries, telemetry).

All HTTP is served by httpx.MockTransport; no network or API key needed.
"""
from __future__ import annotations

import json
import threading

import httpx
import pytest

from jobpilot.voice import (
    AudioCache,
    ElevenLabsClient,
    VoiceConfig,
    VoiceError,
    VoiceInterviewer,
    VoiceMetric,
    clean_for_speech,
    feedback_script,
    question_script,
    summarize,
)
from jobpilot.voice.telemetry import percentile

FAKE_MP3 = b"ID3" + b"\x00" * 64


class FakeElevenLabs:
    """Minimal stand-in for the ElevenLabs API with failure injection."""

    def __init__(self, fail_first: int = 0, fail_status: int = 429, stt_text: str = "I owned the incident."):
        self.fail_first = fail_first
        self.fail_status = fail_status
        self.stt_text = stt_text
        self.tts_calls = 0
        self.stt_calls = 0
        self.requests: list[httpx.Request] = []
        self._lock = threading.Lock()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        with self._lock:
            self.requests.append(request)
            if request.url.path.startswith("/v1/text-to-speech/"):
                self.tts_calls += 1
                n = self.tts_calls
            else:
                self.stt_calls += 1
                n = self.stt_calls
        assert request.headers["xi-api-key"] == "test-key"
        if n <= self.fail_first:
            return httpx.Response(self.fail_status, json={"detail": "busy"}, headers={"retry-after": "0"})
        if request.url.path.startswith("/v1/text-to-speech/"):
            body = json.loads(request.content)
            assert body["model_id"] == "eleven_flash_v2_5"
            assert "voice_settings" in body
            assert request.url.params["output_format"] == "mp3_44100_128"
            return httpx.Response(200, content=FAKE_MP3, headers={"content-type": "audio/mpeg"})
        if request.url.path == "/v1/speech-to-text":
            assert b"scribe_v1" in request.content
            return httpx.Response(200, json={"text": f"  {self.stt_text}  ", "language_code": "en"})
        return httpx.Response(404)


@pytest.fixture
def cfg(tmp_path):
    return VoiceConfig(api_key="test-key", cache_dir=tmp_path / "cache", max_retries=3)


def make_client(cfg, fake, metrics=None):
    sink = (metrics.append if metrics is not None else None)
    return ElevenLabsClient(cfg, transport=httpx.MockTransport(fake), sink=sink, sleep=lambda _s: None)


# ---------------------------------------------------------------- client

def test_tts_returns_audio_and_hits_stream_endpoint(cfg):
    fake = FakeElevenLabs()
    client = make_client(cfg, fake)
    audio = client.synthesize("Walk me through a Kafka outage.")
    assert audio == FAKE_MP3
    assert fake.requests[0].url.path == f"/v1/text-to-speech/{cfg.voice_id}/stream"


def test_tts_second_call_is_cache_hit(cfg):
    fake, metrics = FakeElevenLabs(), []
    client = make_client(cfg, fake, metrics)
    client.synthesize("Same question")
    client.synthesize("  Same   question ")  # whitespace-normalized key
    assert fake.tts_calls == 1
    assert [m.cache_hit for m in metrics] == [False, True]


def test_cache_key_changes_with_voice(cfg):
    a = AudioCache.key("hi", model="m", voice_id="v1", output_format="f", settings={})
    b = AudioCache.key("hi", model="m", voice_id="v2", output_format="f", settings={})
    assert a != b


def test_tts_retries_on_429_then_succeeds(cfg):
    fake, metrics = FakeElevenLabs(fail_first=2, fail_status=429), []
    client = make_client(cfg, fake, metrics)
    assert client.synthesize("retry me") == FAKE_MP3
    assert fake.tts_calls == 3
    assert metrics[-1].attempts == 3


def test_tts_does_not_retry_on_401(cfg):
    fake, metrics = FakeElevenLabs(fail_first=5, fail_status=401), []
    client = make_client(cfg, fake, metrics)
    with pytest.raises(VoiceError) as exc:
        client.synthesize("bad key")
    assert exc.value.status == 401
    assert fake.tts_calls == 1
    assert metrics[-1].status == "error:401"


def test_tts_gives_up_after_max_retries(cfg):
    fake = FakeElevenLabs(fail_first=99, fail_status=503)
    client = make_client(cfg, fake)
    with pytest.raises(VoiceError):
        client.synthesize("always down")
    assert fake.tts_calls == cfg.max_retries + 1


def test_stt_returns_stripped_transcript(cfg):
    fake, metrics = FakeElevenLabs(stt_text="I rolled back the deploy."), []
    client = make_client(cfg, fake, metrics)
    assert client.transcribe(b"RIFF....", filename="a.wav") == "I rolled back the deploy."
    assert metrics[-1].op == "stt"


def test_client_requires_key():
    with pytest.raises(VoiceError):
        ElevenLabsClient(VoiceConfig(api_key=None))


# ----------------------------------------------------------- interviewer

def test_interviewer_disabled_without_key():
    v = VoiceInterviewer(VoiceConfig(api_key=None))
    assert not v.enabled
    assert v.speak("hello") is None
    assert v.transcribe(b"x") is None
    assert v.prefetch(["q1", "q2"]) == {"ok": 0, "failed": 0}


def test_prefetch_warms_cache_concurrently(cfg):
    fake = FakeElevenLabs()
    v = VoiceInterviewer(cfg, client=make_client(cfg, fake))
    qs = ["What is a pod?", "Explain RBAC.", "Tell me about a migration.", "How do you size Kafka?"]
    job = {"title": "Infrastructure Engineer", "company": "ElevenLabs"}
    assert v.prefetch(qs, job) == {"ok": 4, "failed": 0}
    calls = fake.tts_calls
    for i, q in enumerate(qs):
        assert v.speak_question(q, i, len(qs), job) == FAKE_MP3
    assert fake.tts_calls == calls  # every question served from cache


def test_interviewer_falls_back_on_api_failure(cfg):
    fake = FakeElevenLabs(fail_first=99, fail_status=500)
    v = VoiceInterviewer(cfg, client=make_client(cfg, fake))
    assert v.speak("anything") is None
    assert "500" in (v.last_error or "")


# ----------------------------------------------------------- scripts

def test_question_script_intro_and_last():
    job = {"title": "Infrastructure Engineer", "company": "ElevenLabs"}
    first = question_script("What is **etcd**?", 0, 3, job)
    assert first.startswith("Hi, I'll be your interviewer")
    assert "ElevenLabs" in first and "**" not in first
    assert question_script("Q", 1, 3).startswith("Question 2.")
    assert question_script("Q", 2, 3).startswith("Last question.")


def test_clean_for_speech_strips_markdown_and_truncates():
    text = "Use `kubectl` and see [docs](https://k8s.io). " + "word " * 300
    out = clean_for_speech(text, limit=120)
    assert "`" not in out and "https" not in out and "docs" in out
    assert len(out) <= 121


def test_feedback_script():
    s = feedback_script({"overall_score": 7.4, "strengths": ["clear STAR"], "weaknesses": ["no metrics"]})
    assert s == "That scored 7 out of 10. What worked: clear STAR. To improve: no metrics."


# ----------------------------------------------------------- telemetry

def test_percentile():
    assert percentile([10, 20, 30, 40], 50) == 25
    assert percentile([], 50) is None


def test_summarize_ignores_cache_hits_for_latency():
    ms = [
        VoiceMetric(op="tts", model="m", chars=10, ttfb_ms=200.0),
        VoiceMetric(op="tts", model="m", chars=10, ttfb_ms=400.0),
        VoiceMetric(op="tts", model="m", chars=10, ttfb_ms=0.0, cache_hit=True),
        VoiceMetric(op="stt", model="s", total_ms=900.0),
    ]
    s = summarize(ms)
    assert s["tts_ttfb_p50_ms"] == 300.0
    assert s["tts_cache_hit_rate"] == round(1 / 3, 3)
    assert s["tts_chars_billed"] == 20
    assert s["stt_calls"] == 1


def test_db_sink_roundtrip(tmp_path):
    from jobpilot.voice.telemetry import db_sink, load_metrics

    url = f"sqlite:///{tmp_path/'t.db'}"
    sink = db_sink(url)
    sink(VoiceMetric(op="tts", model="eleven_flash_v2_5", chars=42, ttfb_ms=180.0, session_id="s1"))
    rows = load_metrics(url)
    assert len(rows) == 1 and rows[0].chars == 42 and rows[0].session_id == "s1"
