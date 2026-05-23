from __future__ import annotations

import json
import os
import re
import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class TailoredResume:
    header: str
    bullets: List[str]
    injected_keywords: List[str]

    def to_text(self) -> str:
        body = "\n".join(f"- {bullet}" for bullet in self.bullets)
        return f"{self.header}\n\n{body}".strip()


class ResumeTailor:
    """Deterministic resume tailoring by re-ranking bullets against JD keywords."""

    def __init__(self, max_keywords: int = 12):
        self.max_keywords = max_keywords
        self.max_injections = 1
        # low-value tokens we don't want injected
        self.denylist = set([
            "data",
            "experience",
            "working",
            "work",
            "using",
            "team",
            "responsible",
            "responsibilities",
            "skills",
        ])
        # allow a few important short tokens (SQL, AWS, R, C++)
        self.short_whitelist = set(["sql", "c++", "go", "r", "api", "aws", "gcp", "etl", "ml", "nlp", "ai"])
        # preferred casing for common tokens
        self.preferred_casing = {
            "sql": "SQL",
            "c++": "C++",
            "r": "R",
            "api": "API",
            "apis": "APIs",
            "aws": "AWS",
            "gcp": "GCP",
            "etl": "ETL",
            "ml": "ML",
            "nlp": "NLP",
            "ai": "AI",
            "python": "Python",
        }

    def tailor(self, resume_text: str, job: Dict[str, Any]) -> TailoredResume:
        header, bullets = _split_resume(resume_text)
        if not bullets:
            bullets = [line.strip() for line in resume_text.splitlines() if line.strip()]

        jd_keywords = self._extract_keywords(job)
        scored_bullets: List[Tuple[int, str]] = []
        injected: List[str] = []
        used_injections: set[str] = set()
        injections_used = 0

        # also prepare description text/tokens for stricter injection checks
        description_text = str(job.get("description", "")).lower()
        description_tokens = set(_tokenize(description_text))

        for bullet in bullets:
            base_score = _keyword_overlap_score(bullet, jd_keywords)
            improved_bullet, added = bullet, ""
            if injections_used < self.max_injections:
                improved_bullet, added = self._inject_keyword_if_needed(
                    bullet,
                    jd_keywords,
                    description_text,
                    description_tokens,
                    used_injections,
                )
            # normalize bullet to single-line, collapse whitespace
            improved_bullet = _normalize_text(improved_bullet)
            bonus = 1 if added else 0
            scored_bullets.append((base_score + bonus, improved_bullet))
            if added:
                injected.append(added)
                used_injections.add(added)
                injections_used += 1

        scored_bullets.sort(key=lambda pair: pair[0], reverse=True)
        ranked_bullets = [bullet for _, bullet in scored_bullets]

        return TailoredResume(
            header=header,
            bullets=ranked_bullets,
            injected_keywords=sorted(set(injected)),
        )

    def _extract_keywords(self, job: Dict[str, Any]) -> List[Tuple[str, str]]:
        """Return list of (keyword, source) tuples ordered by preference.

        Sources: 'skill', 'responsibility', 'title', 'desc'
        """
        title = str(job.get("title", ""))
        skills = job.get("skills", []) or []
        responsibilities = job.get("responsibilities", []) or []
        description = str(job.get("description", ""))

        keywords: List[Tuple[str, str]] = []
        seen = set()

        # Add whole skill phrases first
        for skill in skills:
            k = str(skill).strip().lower()
            if k and k not in seen:
                keywords.append((k, "skill"))
                seen.add(k)
                if len(keywords) >= self.max_keywords:
                    return keywords

        # Add whole responsibilities
        for resp in responsibilities:
            k = str(resp).strip().lower()
            if k and k not in seen:
                keywords.append((k, "responsibility"))
                seen.add(k)
                if len(keywords) >= self.max_keywords:
                    return keywords

        # Title tokens
        for token in _tokenize(title):
            if token not in seen:
                keywords.append((token, "title"))
                seen.add(token)
                if len(keywords) >= self.max_keywords:
                    return keywords

        # Description tokens (lowest priority)
        for token in _tokenize(description):
            if token not in seen:
                keywords.append((token, "desc"))
                seen.add(token)
                if len(keywords) >= self.max_keywords:
                    return keywords

        return keywords

    def _inject_keyword_if_needed(
        self,
        bullet: str,
        keywords: List[Tuple[str, str]],
        description_text: str,
        description_tokens: set,
        used_injections: set[str],
    ) -> Tuple[str, str]:
        lower_bullet = bullet.lower()
        for keyword, source in keywords:
            if keyword in lower_bullet:
                continue
            if keyword in used_injections:
                continue
            if keyword in self.denylist:
                continue
            # ignore very short tokens unless explicitly whitelisted
            if len(keyword) < 4 and keyword not in self.short_whitelist:
                continue
            # Only inject keywords that come from skills/responsibilities/title
            if source not in ("skill", "responsibility", "title"):
                continue
            # Stricter rule: prefer injecting only if keyword also appears in job description (phrase-aware)
            if not _keyword_in_description(keyword, description_text, description_tokens) and keyword not in self.short_whitelist:
                continue
            # If source is title and keyword is short/non-phrase, skip
            if source == "title" and (len(keyword) < 4 and keyword not in self.short_whitelist):
                continue
            clean_keyword = keyword.strip()
            pretty = self.preferred_casing.get(clean_keyword, None)
            if not pretty:
                # title-case acronyms poorly, so prefer upper for known short tokens
                pretty = clean_keyword.upper() if clean_keyword in self.short_whitelist else clean_keyword.title()

            # phrasing: use 'using' for short technical tokens, 'with' otherwise
            phrase = "using" if clean_keyword in self.short_whitelist or pretty.isupper() else "with"
            if bullet.endswith("."):
                return f"{bullet[:-1]} {phrase} {pretty}.", keyword
            return f"{bullet} {phrase} {pretty}", keyword
        return bullet, ""


def _split_resume(text: str) -> Tuple[str, List[str]]:
    lines = [line.rstrip() for line in text.splitlines()]
    header_lines: List[str] = []
    bullets: List[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("-", "*", "•")):
            bullets.append(stripped.lstrip("-*• ").strip())
        else:
            if bullets:
                bullets.append(stripped)
            else:
                header_lines.append(stripped)

    header = "\n".join(header_lines).strip()
    return header, [bullet for bullet in bullets if bullet]


def _normalize_text(text: str) -> str:
    # collapse any whitespace (including newlines) into single spaces and strip
    return re.sub(r"\s+", " ", str(text)).strip()


def _tokenize(text: str) -> List[str]:
    stopwords = {
        "and",
        "the",
        "for",
        "with",
        "from",
        "that",
        "this",
        "your",
        "role",
        "team",
        "work",
        "will",
        "you",
        "our",
        "data",
        "science",
        "scientist",
        "experience",
        "required",
    }
    raw = re.findall(r"[a-zA-Z][a-zA-Z0-9+#.-]{1,}", text.lower())
    return [token for token in raw if token not in stopwords]


def _keyword_overlap_score(text: str, keywords: List[Tuple[str, str]]) -> int:
    lower = text.lower()
    return sum(1 for keyword, _ in keywords if keyword in lower)


def _keyword_in_description(keyword: str, description_text: str, description_tokens: set) -> bool:
    k = keyword.lower()
    # direct substring match handles multi-word phrases
    if k in description_text:
        return True
    # token-level inclusion (all words present)
    words = re.findall(r"[a-zA-Z0-9+#.-]+", k)
    if words and all(w in description_tokens for w in words):
        return True
    # fuzzy match: compare keyword against each description token
    for token in description_tokens:
        ratio = difflib.SequenceMatcher(None, k, token).ratio()
        if ratio >= 0.85:
            return True
    return False


# ---------------------------------------------------------------------------
# W4: LLM-powered LaTeX resume tailoring
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an expert resume writer specialising in ATS-optimised resumes for data science and ML engineering roles.

Your task: rewrite resume bullet points so they better match a job description.

STRICT RULES:
1. Do NOT invent metrics, tools, or experiences not present in the original bullet.
2. Reframe existing achievements using JD vocabulary and keywords naturally.
3. Keep \\textbf{} LaTeX formatting for numbers/metrics exactly as they appear.
4. Each rewritten bullet must be roughly the same length as the original (±20%).
5. Output ONLY a JSON object: {"bullets": ["bullet1", "bullet2", ...]}
6. The bullets array must have EXACTLY the same number of items as the input."""


def _extract_resume_items(tex: str) -> List[Tuple[int, int, str]]:
    """Return list of (content_start, content_end, content) for every \\resumeItem{...}.

    Uses manual brace-matching so nested \\textbf{} etc. are handled correctly.
    Education \\resumeItem is skipped (it's the coursework line — keep as-is).
    """
    results: List[Tuple[int, int, str]] = []
    marker = r"\resumeItem{"
    # Find the Education section boundary so we skip its item
    edu_end = _section_end(tex, "Education")
    i = 0
    while True:
        pos = tex.find(marker, i)
        if pos == -1:
            break
        # Skip items inside the Education section
        if pos < edu_end:
            i = pos + 1
            continue
        brace_open = pos + len(marker) - 1  # index of '{'
        depth = 0
        j = brace_open
        while j < len(tex):
            if tex[j] == "{":
                depth += 1
            elif tex[j] == "}":
                depth -= 1
                if depth == 0:
                    results.append((brace_open + 1, j, tex[brace_open + 1 : j]))
                    break
            j += 1
        i = pos + 1
    return results


def _section_end(tex: str, section_name: str) -> int:
    """Return the character position where the named section ends (next \\section or EOF)."""
    start = tex.find(f"\\section{{{section_name}}}")
    if start == -1:
        return 0
    next_sec = tex.find(r"\section{", start + 1)
    return next_sec if next_sec != -1 else len(tex)


def _inject_bullets(tex: str, items: List[Tuple[int, int, str]], rewritten: List[str]) -> str:
    """Replace each item's content slice with the rewritten text (reverse order to preserve offsets)."""
    for (start, end, _), new_text in zip(reversed(items), reversed(rewritten)):
        tex = tex[:start] + new_text + tex[end:]
    return tex


def _extract_skills_block(tex: str) -> Optional[Tuple[int, int, str]]:
    """Return (start, end, text) of the \\item{...} content inside Technical Skills."""
    # Locate the \small{\item{ ... }} block
    sec_start = tex.find(r"\section{Technical Skills}")
    if sec_start == -1:
        return None
    sec_end = _section_end(tex, "Technical Skills")
    chunk = tex[sec_start:sec_end]

    marker = r"\small{\item{"
    pos = chunk.find(marker)
    if pos == -1:
        return None
    open_pos = pos + len(marker) - 1  # index of last {
    abs_open = sec_start + open_pos

    depth = 0
    j = abs_open
    while j < sec_start + len(chunk):
        if tex[j] == "{":
            depth += 1
        elif tex[j] == "}":
            depth -= 1
            if depth == 0:
                return (abs_open + 1, j, tex[abs_open + 1 : j])
        j += 1
    return None


class LLMResumeTailor:
    """GPT-4o-mini resume tailoring for LaTeX templates.

    Rewrites \\resumeItem{} bullets in Projects + Experience sections to match
    the target JD, then reorders the Technical Skills categories by relevance.
    Falls back to returning the template unchanged when no API key is set or
    the LLM call fails.

    [INTERVIEW GOLD] Two-pass approach: (1) extract bullet texts via brace-matching
    so LaTeX structure is never touched by the LLM, (2) inject rewritten text back
    at exact char offsets. The LLM only sees/writes plain + \\textbf{} text, not
    document structure — this prevents hallucinated LaTeX breaking compilation.
    """

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = None

    @property
    def _llm(self):
        if self._client is None and self._api_key:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def tailor_latex(self, template_tex: str, job: Dict[str, Any]) -> str:
        """Return a copy of template_tex with bullets + skills rewritten for this job."""
        if not self._llm:
            return template_tex  # no key → return unchanged

        items = _extract_resume_items(template_tex)
        if not items:
            return template_tex

        try:
            rewritten = self._rewrite_bullets([content for _, _, content in items], job)
            tex = _inject_bullets(template_tex, items, rewritten)
            tex = self._reorder_skills(tex, job)
            return tex
        except Exception:
            return template_tex  # any LLM failure → unchanged template

    def _rewrite_bullets(self, bullets: List[str], job: Dict[str, Any]) -> List[str]:
        jd_summary = (
            f"Title: {job.get('title', '')}\n"
            f"Company: {job.get('company', '')}\n"
            f"Key skills: {', '.join(str(s) for s in (job.get('skills') or []))}\n"
            f"Description (truncated): {str(job.get('description', ''))[:1500]}"
        )
        prompt = (
            f"JOB DESCRIPTION:\n{jd_summary}\n\n"
            f"BULLETS TO REWRITE (LaTeX, keep \\textbf{{}} for metrics):\n"
            f"{json.dumps(bullets, indent=2)}\n\n"
            "Output JSON: {\"bullets\": [\"rewritten1\", ...]}"
        )
        resp = self._llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        parsed = json.loads(resp.choices[0].message.content)
        rewritten: List[str] = parsed.get("bullets", [])
        # Safety: must match original count exactly
        if len(rewritten) != len(bullets):
            return bullets
        return [str(b) for b in rewritten]

    def _reorder_skills(self, tex: str, job: Dict[str, Any]) -> str:
        """Reorder skill category lines so the most JD-relevant appear first."""
        jd_skills = ", ".join(str(s) for s in (job.get("skills") or []))
        if not jd_skills or not self._llm:
            return tex

        block = _extract_skills_block(tex)
        if not block:
            return tex
        start, end, skills_text = block

        prompt = (
            f"Job requires: {jd_skills}\n\n"
            f"Current skills block (LaTeX):\n{skills_text}\n\n"
            "Reorder the \\textbf{{Category:}} lines so the most relevant category "
            "appears first. Keep every word inside each category EXACTLY the same. "
            "Return ONLY the reordered block, no explanation."
        )
        try:
            resp = self._llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            new_block = resp.choices[0].message.content.strip()
            # Sanity: must still contain at least two \\textbf{ markers
            if new_block.count(r"\textbf{") >= 2:
                return tex[:start] + new_block + tex[end:]
        except Exception:
            pass
        return tex
