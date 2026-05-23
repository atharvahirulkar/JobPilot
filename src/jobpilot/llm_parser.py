from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict
from urllib import error, request

from .jd_parser import JDParser


@dataclass
class LLMConfig:
    provider: str = os.getenv("JOBPILOT_LLM_PROVIDER", "local")
    model: str = os.getenv("JOBPILOT_LLM_MODEL", "llama3.1:8b")
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")


class LLMJDParser:
    """Structured JD parser backed by a local model first, with deterministic fallback.

    The parser asks for JSON only. If the local LLM is unavailable or returns
    invalid output, it falls back to the heuristic parser so the pipeline stays
    usable offline.
    """

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        self.fallback = JDParser()

    def parse(self, text: str) -> Dict[str, Any]:
        if self.config.provider == "local":
            parsed = self._parse_with_ollama(text)
            if parsed is not None:
                return parsed
        return self.fallback.parse(text)

    def _parse_with_ollama(self, text: str) -> Dict[str, Any] | None:
        prompt = (
            "Extract a job description into strict JSON with keys: "
            "title, company, location, seniority, skills, responsibilities, summary. "
            "skills and responsibilities must be arrays of short strings. "
            "Return JSON only, no markdown, no commentary.\n\n"
            f"JOB DESCRIPTION:\n{text}"
        )
        payload = {
            "model": self.config.model,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": "You extract structured fields from job descriptions."},
                {"role": "user", "content": prompt},
            ],
        }

        url = f"{self.config.ollama_url.rstrip('/')}/api/chat"
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")

        try:
            with request.urlopen(req, timeout=30) as resp:
                body = resp.read().decode("utf-8")
        except (error.URLError, TimeoutError, ConnectionError, OSError):
            return None

        try:
            outer = json.loads(body)
            content = outer["message"]["content"]
            parsed = json.loads(content)
        except (KeyError, TypeError, json.JSONDecodeError):
            return None

        if not isinstance(parsed, dict):
            return None

        parsed.setdefault("summary", "")
        parsed.setdefault("skills", [])
        parsed.setdefault("responsibilities", [])
        return parsed


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Demo LLM-backed JD parser")
    parser.add_argument("jdfile", help="Path to JD text file")
    args = parser.parse_args()
    text = Path(args.jdfile).read_text()
    parsed = LLMJDParser().parse(text)
    print(json.dumps(parsed, indent=2))
