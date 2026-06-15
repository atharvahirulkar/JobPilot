"""Company-targeted job scraper — career page URLs + Playwright + ATS APIs.

Reads your curated company list (data/🏢 Company.csv), visits each company's
Career Page, detects embedded ATS boards (Greenhouse, Lever, Ashby, Workable),
falls back to HTML link extraction, then optional OpenAI parsing.

Rotating cursor
---------------
~450 companies per full scan is slow (Playwright).  We process `batch_size`
companies per run and advance a cursor in `data/search_cursor.json`.
Default batch of 30 → full cycle in ~15 days at one run/day.  Cost: $0 for
fetching; optional OpenAI only when ATS + HTML extraction both fail.

No SerpAPI required.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .ats_fetchers import (
    detect_ats_from_html,
    detect_ats_from_url,
    extract_jobs_from_html,
    fetch_ats_jobs,
    parse_jobs_with_llm,
)
from .job_normalize import dedupe_jobs, normalize_job

log = logging.getLogger("jobpilot.company_scraper")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_COMPANIES_CSV = Path("data/🏢 Company.csv")
_CURSOR_FILE           = Path("data/search_cursor.json")
_DEFAULT_BATCH_SIZE    = 30
_INTER_COMPANY_DELAY   = 1.0   # seconds between companies (be polite)

_DEFAULT_ROLES = [
    "Data Scientist",
    "Machine Learning Engineer",
    "Data Engineer",
    "Software Engineer",
    "AI Engineer",
    "Quantitative Researcher",
]

# Keywords used to filter job titles (one career page fetch → filter locally)
_ROLE_KEYWORDS: Dict[str, List[str]] = {
    "Data Scientist":              ["data scientist", "data science"],
    "Machine Learning Engineer":   ["machine learning engineer", "ml engineer", "mle"],
    "Data Engineer":               ["data engineer"],
    "Software Engineer":           ["software engineer", "swe", "backend engineer", "full stack"],
    "AI Engineer":                 ["ai engineer", "artificial intelligence engineer"],
    "Quantitative Researcher":     ["quantitative researcher", "quant researcher", "quant research"],
    "Quantitative Analyst":        ["quantitative analyst", "quant analyst"],
    "Research Scientist":          ["research scientist"],
    "Applied Scientist":           ["applied scientist"],
    "Analytics Engineer":          ["analytics engineer"],
    "MLOps Engineer":              ["mlops", "ml ops engineer"],
    "NLP Engineer":                ["nlp engineer", "natural language"],
    "Computer Vision Engineer":    ["computer vision", "cv engineer"],
    "Business Intelligence Engineer": ["business intelligence", "bi engineer"],
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class CompanyScraper:
    """Scrape career pages from your target-company list."""

    def __init__(
        self,
        companies_csv:     Path | str    = _DEFAULT_COMPANIES_CSV,
        cursor_file:       Path | str    = _CURSOR_FILE,
        roles:             List[str]     = None,
        batch_size:        int           = _DEFAULT_BATCH_SIZE,
        use_llm_fallback:  bool          = True,
        api_key:           Optional[str] = None,
    ):
        self.companies_csv    = Path(companies_csv)
        self.cursor_file      = Path(cursor_file)
        self.roles            = roles or _DEFAULT_ROLES
        self.batch_size       = batch_size
        self.use_llm_fallback = use_llm_fallback
        self.api_key          = api_key or os.getenv("OPENAI_API_KEY")

    def scrape(
        self,
        progress_cb: Optional[Callable[[str, int, int], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Return de-duped normalised job dicts for the current company batch."""
        companies = self._load_companies()
        if not companies:
            log.warning("No companies loaded from %s", self.companies_csv)
            return []

        batch = self._get_batch(companies)
        total = len(batch)
        all_raw: List[Dict[str, Any]] = []

        log.info(
            "Scraping %d companies (batch %d–%d of %d total)",
            total,
            self._cursor_index(),
            self._cursor_index() + total - 1,
            len(companies),
        )

        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            log.error("playwright not installed — run: pip install playwright && playwright install chromium")
            return []

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            try:
                for i, company in enumerate(batch):
                    name = company["company"]
                    url  = company.get("url", "")
                    if progress_cb:
                        progress_cb(f"Scraping: {name}", i + 1, total)

                    if not url:
                        log.debug("Skipping %s — no career page URL", name)
                        continue

                    try:
                        jobs = self._scrape_one(browser, name, url)
                        filtered = _filter_by_roles(jobs, self.roles)
                        all_raw.extend(filtered)
                        log.info("%s: %d jobs (%d after role filter)", name, len(jobs), len(filtered))
                    except Exception as exc:
                        log.warning("Scrape failed for %s: %s", name, exc)

                    time.sleep(_INTER_COMPANY_DELAY)
            finally:
                browser.close()

        self._advance_cursor(len(companies))

        normalised = []
        for raw in all_raw:
            try:
                normalised.append(normalize_job(raw).to_dict())
            except Exception as exc:
                log.debug("Normalisation failed: %s", exc)

        deduped = dedupe_jobs(normalised)
        log.info("Scraped %d raw → %d deduped jobs", len(all_raw), len(deduped))
        return deduped

    def _scrape_one(self, browser, company_name: str, career_url: str) -> List[Dict[str, Any]]:
        # 1. Direct ATS from career URL
        target = detect_ats_from_url(career_url)
        if target:
            jobs = fetch_ats_jobs(target, company_name)
            if jobs:
                return jobs

        # 2. Fetch page HTML via Playwright
        html = _fetch_html(browser, career_url)
        if not html:
            return []

        # 3. Embedded ATS in page
        target = detect_ats_from_html(html)
        if target:
            jobs = fetch_ats_jobs(target, company_name)
            if jobs:
                return jobs

        # 4. Link extraction from HTML
        jobs = extract_jobs_from_html(html, company_name, career_url)
        if jobs:
            return jobs

        # 5. Optional OpenAI parse
        if self.use_llm_fallback and self.api_key:
            return parse_jobs_with_llm(html, company_name, career_url, self.api_key)

        return []

    # ── CSV / cursor helpers ────────────────────────────────────────────────

    def _load_companies(self) -> List[Dict[str, str]]:
        if not self.companies_csv.exists():
            log.error("Company CSV not found: %s", self.companies_csv)
            return []
        try:
            with self.companies_csv.open(encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            result = []
            for row in rows:
                keys = {k.strip().lower(): (v or "").strip() for k, v in row.items()}
                name = keys.get("company") or keys.get("name") or ""
                url  = keys.get("career page") or keys.get("url") or keys.get("careers") or ""
                if name:
                    result.append({"company": name, "url": url})
            return result
        except Exception as exc:
            log.error("Failed to load companies CSV: %s", exc)
            return []

    def _cursor_index(self) -> int:
        if self.cursor_file.exists():
            try:
                return int(json.loads(self.cursor_file.read_text()).get("cursor", 0))
            except Exception:
                pass
        return 0

    def _get_batch(self, companies: List[Dict]) -> List[Dict]:
        if not companies:
            return []
        start = self._cursor_index() % len(companies)
        batch = companies[start: start + self.batch_size]
        if len(batch) < self.batch_size:
            batch += companies[: self.batch_size - len(batch)]
        return batch[: self.batch_size]

    def _advance_cursor(self, total_companies: int) -> None:
        next_cursor = (self._cursor_index() + self.batch_size) % total_companies
        self.cursor_file.parent.mkdir(parents=True, exist_ok=True)
        self.cursor_file.write_text(json.dumps({
            "cursor":    next_cursor,
            "last_run":  datetime.utcnow().isoformat(),
            "last_date": date.today().isoformat(),
        }))

    def cursor_status(self) -> Dict[str, Any]:
        companies = self._load_companies()
        total     = len(companies)
        cursor    = self._cursor_index()
        with_url  = sum(1 for c in companies if c.get("url"))

        meta: Dict[str, Any] = {
            "total_companies":    total,
            "companies_with_url": with_url,
            "cursor":             cursor,
            "batch_size":         self.batch_size,
            "next_batch_preview": [c["company"] for c in self._get_batch(companies)[:5]],
            "days_to_full_cycle": round(total / self.batch_size, 1) if self.batch_size else 0,
            "last_run":           None,
            "last_date":          None,
            "scraper_mode":       "playwright+ats",
        }
        if self.cursor_file.exists():
            try:
                data = json.loads(self.cursor_file.read_text())
                meta["last_run"]  = data.get("last_run")
                meta["last_date"] = data.get("last_date")
            except Exception:
                pass
        return meta


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _fetch_html(browser, url: str, timeout_ms: int = 30000) -> str:
    page = browser.new_page()
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        page.wait_for_timeout(2000)
        return page.content()
    except Exception as exc:
        log.warning("Playwright fetch failed for %s: %s", url, exc)
        return ""
    finally:
        page.close()


def _filter_by_roles(jobs: List[Dict[str, Any]], roles: List[str]) -> List[Dict[str, Any]]:
    if not roles:
        return jobs
    keywords: List[str] = []
    for role in roles:
        keywords.extend(_ROLE_KEYWORDS.get(role, [role.lower()]))
    keywords = list(dict.fromkeys(keywords))

    filtered = []
    for job in jobs:
        title = (job.get("title") or "").lower()
        if any(kw in title for kw in keywords):
            filtered.append(job)
    return filtered


def scrape_company_jobs(
    batch_size:  int                            = _DEFAULT_BATCH_SIZE,
    roles:       List[str]                      = None,
    progress_cb: Optional[Callable]             = None,
) -> List[Dict[str, Any]]:
    """Top-level helper called from run_pipeline()."""
    return CompanyScraper(
        batch_size=batch_size,
        roles=roles or _DEFAULT_ROLES,
    ).scrape(progress_cb=progress_cb)
