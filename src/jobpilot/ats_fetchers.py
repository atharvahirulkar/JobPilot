"""ATS-aware career page fetchers — free JSON APIs for common job boards.

Supported providers (detected from URL or embedded in page HTML):
  - Greenhouse  boards-api.greenhouse.io
  - Lever       api.lever.co
  - Ashby       api.ashbyhq.com
  - Workable    apply.workable.com/api
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

log = logging.getLogger("jobpilot.ats_fetchers")

_USER_AGENT = "JobPilot/1.0 (+local career scraper)"
_REQUEST_TIMEOUT = 20


@dataclass(frozen=True)
class ATSTarget:
    provider: str   # greenhouse | lever | ashby | workable
    token: str      # board slug / company id


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

_URL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("greenhouse", re.compile(r"boards\.greenhouse\.io/(?P<token>[^/?#\s\"']+)", re.I)),
    ("greenhouse", re.compile(r"job-boards\.greenhouse\.io/(?P<token>[^/?#\s\"']+)", re.I)),
    ("lever",      re.compile(r"jobs\.lever\.co/(?P<token>[^/?#\s\"']+)", re.I)),
    ("ashby",      re.compile(r"jobs\.ashbyhq\.com/(?P<token>[^/?#\s\"']+)", re.I)),
    ("workable",   re.compile(r"apply\.workable\.com/(?P<token>[^/?#\s\"']+)", re.I)),
]


def detect_ats_from_url(url: str) -> Optional[ATSTarget]:
    for provider, pattern in _URL_PATTERNS:
        m = pattern.search(url or "")
        if m:
            return ATSTarget(provider=provider, token=m.group("token"))
    return None


def detect_ats_from_html(html: str) -> Optional[ATSTarget]:
    for provider, pattern in _URL_PATTERNS:
        m = pattern.search(html or "")
        if m:
            return ATSTarget(provider=provider, token=m.group("token"))
    return None


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_ats_jobs(target: ATSTarget, company_name: str) -> List[Dict[str, Any]]:
    fetchers = {
        "greenhouse": _fetch_greenhouse,
        "lever":      _fetch_lever,
        "ashby":      _fetch_ashby,
        "workable":   _fetch_workable,
    }
    fn = fetchers.get(target.provider)
    if not fn:
        return []
    try:
        return fn(target.token, company_name)
    except Exception as exc:
        log.warning("ATS fetch failed (%s/%s): %s", target.provider, target.token, exc)
        return []


def _http_get_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _fetch_greenhouse(token: str, company_name: str) -> List[Dict[str, Any]]:
    data = _http_get_json(
        f"https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true"
    )
    jobs = []
    for item in data.get("jobs", []):
        loc = item.get("location") or {}
        location = loc.get("name", "") if isinstance(loc, dict) else str(loc)
        jobs.append({
            "title":       item.get("title", ""),
            "company":     company_name,
            "location":    location,
            "description": item.get("content", "") or "",
            "url":         item.get("absolute_url", ""),
            "source":      "greenhouse",
        })
    return jobs


def _fetch_lever(token: str, company_name: str) -> List[Dict[str, Any]]:
    data = _http_get_json(f"https://api.lever.co/v0/postings/{token}?mode=json")
    if not isinstance(data, list):
        return []
    jobs = []
    for item in data:
        cats = item.get("categories") or {}
        location = cats.get("location", "") if isinstance(cats, dict) else ""
        jobs.append({
            "title":       item.get("text", ""),
            "company":     company_name,
            "location":    location,
            "description": item.get("descriptionPlain", "") or item.get("description", "") or "",
            "url":         item.get("hostedUrl", "") or item.get("applyUrl", ""),
            "source":      "lever",
        })
    return jobs


def _fetch_ashby(token: str, company_name: str) -> List[Dict[str, Any]]:
    data = _http_get_json(f"https://api.ashbyhq.com/posting-api/job-board/{token}")
    jobs = []
    for item in data.get("jobs", []):
        jobs.append({
            "title":       item.get("title", ""),
            "company":     company_name,
            "location":    item.get("location", "") or "",
            "description": item.get("descriptionPlain", "") or item.get("description", "") or "",
            "url":         item.get("jobUrl", "") or item.get("applyUrl", ""),
            "source":      "ashby",
        })
    return jobs


def _fetch_workable(token: str, company_name: str) -> List[Dict[str, Any]]:
    data = _http_get_json(f"https://apply.workable.com/api/v1/accounts/{token}/jobs")
    jobs = []
    for item in data.get("jobs", []):
        loc = item.get("location") or {}
        if isinstance(loc, dict):
            location = loc.get("city") or loc.get("country") or ""
            if loc.get("telecommuting"):
                location = f"Remote ({location})".strip(" ()") if location else "Remote"
        else:
            location = str(loc)
        jobs.append({
            "title":       item.get("title", ""),
            "company":     company_name,
            "location":    location,
            "description": item.get("description", "") or "",
            "url":         item.get("url") or item.get("shortlink", ""),
            "source":      "workable",
        })
    return jobs


# ---------------------------------------------------------------------------
# Generic HTML link extraction (no browser, no LLM)
# ---------------------------------------------------------------------------

_JOB_PATH = re.compile(
    r"(?:/jobs?/|/careers?/|/positions?/|/openings?/|/role/|/posting/)",
    re.I,
)
_HREF_RE = re.compile(
    r"""<a[^>]+href=["']([^"']+)["'][^>]*>(.*?)</a>""",
    re.I | re.S,
)
_TAG_RE = re.compile(r"<[^>]+>")


def extract_jobs_from_html(
    html: str,
    company_name: str,
    base_url: str,
) -> List[Dict[str, Any]]:
    """Harvest job-like links from a career page."""
    if not html:
        return []

    seen_urls: set[str] = set()
    jobs: List[Dict[str, Any]] = []

    for href, inner in _HREF_RE.findall(html):
        href = href.strip()
        if not href or href.startswith(("#", "mailto:", "javascript:")):
            continue
        full_url = urljoin(base_url, href)
        if full_url in seen_urls:
            continue
        if not _looks_like_job_url(full_url):
            continue

        title = _TAG_RE.sub(" ", inner)
        title = re.sub(r"\s+", " ", title).strip()
        if not title or len(title) < 4 or len(title) > 120:
            continue
        if _is_noise_title(title):
            continue

        seen_urls.add(full_url)
        jobs.append({
            "title":       title,
            "company":     company_name,
            "location":    "",
            "description": "",
            "url":         full_url,
            "source":      "career_page",
        })

    return jobs


def _looks_like_job_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    if _JOB_PATH.search(path):
        return True
    # e.g. greenhouse job detail URLs
    if re.search(r"/jobs/\d+", path):
        return True
    return False


def _is_noise_title(title: str) -> bool:
    lower = title.lower()
    noise = {
        "careers", "apply now", "view all", "see all", "learn more",
        "join us", "our team", "life at", "benefits", "locations",
        "privacy", "cookie", "sign in", "log in",
    }
    return lower in noise or any(lower.startswith(n) for n in ("view ", "see ", "read "))


# ---------------------------------------------------------------------------
# Optional OpenAI extraction for unstructured pages
# ---------------------------------------------------------------------------

_LLM_SYSTEM = """\
Extract open job postings from career page text.
Return ONLY valid JSON: {"jobs": [{"title": str, "location": str, "url": str, "description": str}]}
Include only real job roles (not nav links like "Careers" or "Benefits").
If no jobs found, return {"jobs": []}.
"""


def parse_jobs_with_llm(
    html: str,
    company_name: str,
    base_url: str,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    key = api_key or __import__("os").getenv("OPENAI_API_KEY")
    if not key:
        return []

    text = _html_to_text(html)[:10000]
    if len(text) < 100:
        return []

    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _LLM_SYSTEM},
                {"role": "user", "content": f"Company: {company_name}\nBase URL: {base_url}\n\nPage text:\n{text}"},
            ],
        )
        raw = json.loads(response.choices[0].message.content or "{}")
        jobs = []
        for item in raw.get("jobs", []):
            url = item.get("url", "")
            if url and not url.startswith("http"):
                url = urljoin(base_url, url)
            jobs.append({
                "title":       item.get("title", ""),
                "company":     company_name,
                "location":    item.get("location", "") or "",
                "description": item.get("description", "") or "",
                "url":         url,
                "source":      "llm_parse",
            })
        return [j for j in jobs if j["title"]]
    except Exception as exc:
        log.warning("LLM job parse failed for %s: %s", company_name, exc)
        return []


def _html_to_text(html: str) -> str:
    text = _TAG_RE.sub(" ", html)
    return re.sub(r"\s+", " ", text).strip()
