"""Voice configuration, loaded from environment variables.

Voice is strictly optional: with no ELEVENLABS_API_KEY the interview runs in
text mode exactly as before, and every voice call becomes a no-op.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class VoiceSettings:
    """ElevenLabs voice_settings. Interviewer defaults: steady, clear, unhurried."""

    stability: float = 0.55
    similarity_boost: float = 0.75
    style: float = 0.0
    use_speaker_boost: bool = True

    def as_payload(self) -> dict:
        return {
            "stability": self.stability,
            "similarity_boost": self.similarity_boost,
            "style": self.style,
            "use_speaker_boost": self.use_speaker_boost,
        }


@dataclass(frozen=True)
class VoiceConfig:
    api_key: str | None = None
    base_url: str = "https://api.elevenlabs.io"
    # Default: "Rachel", a stock ElevenLabs voice available on every account.
    voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    # Flash is ElevenLabs' low-latency TTS model; latency matters more than
    # expressiveness for a back-and-forth interview.
    tts_model: str = "eleven_flash_v2_5"
    stt_model: str = "scribe_v1"
    output_format: str = "mp3_44100_128"
    settings: VoiceSettings = field(default_factory=VoiceSettings)
    cache_dir: Path = Path(".cache/voice")
    cache_max_mb: int = 200
    timeout_s: float = 20.0
    max_retries: int = 3
    prefetch_workers: int = 4

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    @classmethod
    def from_env(cls) -> "VoiceConfig":
        return cls(
            api_key=os.getenv("ELEVENLABS_API_KEY") or None,
            base_url=os.getenv("ELEVENLABS_BASE_URL", cls.base_url),
            voice_id=os.getenv("ELEVENLABS_VOICE_ID", cls.voice_id),
            tts_model=os.getenv("ELEVENLABS_TTS_MODEL", cls.tts_model),
            stt_model=os.getenv("ELEVENLABS_STT_MODEL", cls.stt_model),
            output_format=os.getenv("ELEVENLABS_OUTPUT_FORMAT", cls.output_format),
            settings=VoiceSettings(
                stability=_env_float("ELEVENLABS_STABILITY", 0.55),
                similarity_boost=_env_float("ELEVENLABS_SIMILARITY", 0.75),
            ),
            cache_dir=Path(os.getenv("JOBPILOT_VOICE_CACHE", ".cache/voice")),
            cache_max_mb=_env_int("JOBPILOT_VOICE_CACHE_MB", 200),
            timeout_s=_env_float("ELEVENLABS_TIMEOUT_S", 20.0),
            max_retries=_env_int("ELEVENLABS_MAX_RETRIES", 3),
            prefetch_workers=_env_int("JOBPILOT_VOICE_PREFETCH_WORKERS", 4),
        )
