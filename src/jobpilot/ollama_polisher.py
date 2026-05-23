"""Ollama client wrapper for cover letter polishing."""
from __future__ import annotations

import os
from typing import Optional

try:
    import ollama
except ImportError:
    ollama = None


class OllamaPolisher:
    """Polish cover letters using local Ollama LLM."""

    def __init__(self, host: str = "http://localhost:11434", model: str = "mistral"):
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "mistral")
        self._available = None

    @property
    def available(self) -> bool:
        """Check if Ollama is available (lazy-evaluated once)."""
        if self._available is None:
            if ollama is None:
                self._available = False
            else:
                try:
                    # Try a quick health check
                    response = ollama.list(host=self.host)
                    self._available = response is not None
                except Exception:
                    self._available = False
        return self._available

    def polish_cover_letter(self, cover_letter_text: str) -> Optional[str]:
        """
        Polish a cover letter using Ollama.

        Returns the polished text, or None if Ollama unavailable.
        """
        if not self.available:
            return None

        if ollama is None:
            return None

        prompt = f"""You are an expert cover letter editor. Improve the following cover letter for clarity, 
professionalism, and impact. Keep it concise (3-4 paragraphs). Maintain the key details and structure.

Original cover letter:
{cover_letter_text}

Improved cover letter:"""

        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                host=self.host,
                stream=False,
            )
            polished = response.get("response", "").strip()
            return polished if polished else None
        except Exception:
            return None

    def polish_bullets(self, bullets: list[str], job_title: str = "") -> Optional[list[str]]:
        """
        Polish resume bullets using Ollama.

        Returns list of polished bullets, or None if Ollama unavailable.
        """
        if not self.available or ollama is None:
            return None

        bullet_text = "\n".join(f"- {b}" for b in bullets)
        prompt = f"""You are an expert resume writer. Improve these resume bullets for a {job_title} role.
Make them more impactful, specific, and action-oriented. Keep the same structure (one per line).
Use strong action verbs and quantify impact where possible.

Original bullets:
{bullet_text}

Improved bullets:"""

        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                host=self.host,
                stream=False,
            )
            polished_text = response.get("response", "").strip()
            # Extract bullets from response
            lines = [line.strip().lstrip("- •*") for line in polished_text.split("\n") if line.strip()]
            return lines if lines else None
        except Exception:
            return None


if __name__ == "__main__":
    polisher = OllamaPolisher()
    print(f"Ollama available: {polisher.available}")
    
    if polisher.available:
        test_letter = """Dear Hiring Team,

I am interested in the Data Scientist position. I have experience with Python and data analysis.
I believe I would be a good fit for your team.

Sincerely,
Jane Doe"""
        
        polished = polisher.polish_cover_letter(test_letter)
        if polished:
            print("Polished cover letter:")
            print(polished)
