from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class CoverLetterRequest:
    candidate_name: str
    company: str
    title: str
    resume_highlights: List[str]
    skills: List[str]
    candidate_email: str = ""
    candidate_phone: str = ""
    candidate_linkedin: str = ""
    candidate_portfolio: str = ""


class CoverLetterGenerator:
    """Template-based cover letter generation for predictable, low-detection output."""

    def generate(self, request: CoverLetterRequest) -> str:
        highlights = request.resume_highlights[:2] if request.resume_highlights else ["delivering measurable impact"]
        highlights = [_clean_sentence(item) for item in highlights]
        skills_text = ", ".join(request.skills[:5]) if request.skills else "Python, SQL, and machine learning"

        intro = (
            f"Dear Hiring Team at {request.company},\n\n"
            f"I am excited to apply for the {request.title} role. "
            f"I enjoy solving practical data problems and building production-ready solutions."
        )

        fit = (
            f"In recent work, I focused on {highlights[0]}. "
            f"I also contributed to {highlights[-1]}. "
            f"This aligns well with your needs for strong execution using {skills_text}."
        )

        # Build contact line for signature
        contact_parts = []
        if request.candidate_email:
            contact_parts.append(request.candidate_email)
        if request.candidate_phone:
            contact_parts.append(request.candidate_phone)
        if request.candidate_linkedin:
            contact_parts.append(request.candidate_linkedin)
        if request.candidate_portfolio:
            contact_parts.append(request.candidate_portfolio)
        contact_line = "  |  ".join(contact_parts)

        sig = f"Sincerely,\n{request.candidate_name}"
        if contact_line:
            sig += f"\n{contact_line}"

        close = (
            f"I would value the opportunity to discuss how I can contribute to {request.company} from day one. "
            f"Thank you for your time and consideration.\n\n"
            f"{sig}"
        )

        return "\n\n".join([intro, fit, close]).strip()


def build_request(
    candidate_name: str,
    job: Dict[str, Any],
    tailored_bullets: List[str],
    candidate_email: str = "",
    candidate_phone: str = "",
    candidate_linkedin: str = "",
    candidate_portfolio: str = "",
) -> CoverLetterRequest:
    return CoverLetterRequest(
        candidate_name=candidate_name,
        company=str(job.get("company", "the company")) or "the company",
        title=str(job.get("title", "the role")) or "the role",
        resume_highlights=tailored_bullets,
        skills=[str(item) for item in (job.get("skills", []) or [])],
        candidate_email=candidate_email,
        candidate_phone=candidate_phone,
        candidate_linkedin=candidate_linkedin,
        candidate_portfolio=candidate_portfolio,
    )


def _clean_sentence(text: str) -> str:
    s = str(text)
    # collapse internal whitespace/newlines and strip
    s = " ".join(s.split())
    s = s.strip()
    # remove trailing period to let template control punctuation
    return s.rstrip(".")


# ---------------------------------------------------------------------------
# W4: LLM-powered cover letter generation
# ---------------------------------------------------------------------------

_CL_SYSTEM = """You are an expert cover letter writer for data science and ML engineering job applications.

Write a 3-paragraph cover letter using this structure:
  Para 1 — HOOK: Open with specific excitement about the company/role + your headline value prop
  Para 2 — FIT: 2-3 concrete examples from your background that directly match the JD requirements
  Para 3 — CLOSE: Forward-looking statement + clear call to action

RULES:
1. Do NOT use generic phrases like "I am excited to apply" or "I believe I would be a great fit"
2. Reference specific JD requirements by name (tools, methods, scale)
3. Keep total length to ~200 words across the 3 paragraphs
4. Do NOT add a salutation or signature — those are added separately
5. Output ONLY the 3 paragraphs separated by blank lines, nothing else"""


class LLMCoverLetterGenerator:
    """GPT-4o-mini cover letter generator (hook → fit → close).

    Falls back to template-based CoverLetterGenerator when no API key is set
    or the LLM call fails.

    [INTERVIEW GOLD] Structured prompt with explicit paragraph roles keeps
    the LLM output predictable without post-processing regex hacks.
    """

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = None
        self._fallback = CoverLetterGenerator()

    @property
    def _llm(self):
        if self._client is None and self._api_key:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def generate(self, request: CoverLetterRequest) -> str:
        """Return a 3-paragraph cover letter string (no salutation/signature)."""
        if not self._llm:
            return self._fallback.generate(request)
        try:
            return self._generate_llm(request)
        except Exception:
            return self._fallback.generate(request)

    def _generate_llm(self, request: CoverLetterRequest) -> str:
        skills_text = ", ".join(request.skills[:8]) if request.skills else "Python, SQL, machine learning"
        highlights  = "; ".join(request.resume_highlights[:3]) if request.resume_highlights else ""

        # Build candidate context line (name + any contact info provided)
        candidate_line = request.candidate_name or "the candidate"
        contact_bits = [x for x in [
            request.candidate_email,
            request.candidate_linkedin,
            request.candidate_portfolio,
        ] if x]
        if contact_bits:
            candidate_line += " (" + " | ".join(contact_bits) + ")"

        prompt = (
            f"Candidate: {candidate_line}\n"
            f"Applying for: {request.title} at {request.company}\n"
            f"JD key skills: {skills_text}\n"
            f"Relevant highlights from resume: {highlights}\n\n"
            "Write the 3-paragraph cover letter body (no salutation, no signature)."
        )
        resp = self._llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _CL_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.5,
            max_tokens=400,
        )
        return resp.choices[0].message.content.strip()
