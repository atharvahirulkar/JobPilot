"""Voice mode for JobPilot mock interviews, powered by ElevenLabs.

Text-to-speech reads each question and the feedback aloud; Scribe
speech-to-text turns spoken answers into transcripts for the LLM judge.
"""
from .cache import AudioCache
from .client import ElevenLabsClient, VoiceError
from .config import VoiceConfig, VoiceSettings
from .interviewer import VoiceInterviewer, clean_for_speech, feedback_script, question_script
from .telemetry import VoiceMetric, summarize

__all__ = [
    "AudioCache",
    "ElevenLabsClient",
    "VoiceConfig",
    "VoiceError",
    "VoiceInterviewer",
    "VoiceMetric",
    "VoiceSettings",
    "clean_for_speech",
    "feedback_script",
    "question_script",
    "summarize",
]
