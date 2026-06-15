"""Tests for career-page scraper and ATS fetchers."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from jobpilot.ats_fetchers import (
    ATSTarget,
    detect_ats_from_html,
    detect_ats_from_url,
    extract_jobs_from_html,
    fetch_ats_jobs,
)
from jobpilot.company_scraper import CompanyScraper, _filter_by_roles


# ---------------------------------------------------------------------------
# ATS detection
# ---------------------------------------------------------------------------

def test_detect_ats_from_url_greenhouse():
    t = detect_ats_from_url("https://boards.greenhouse.io/stripe")
    assert t == ATSTarget(provider="greenhouse", token="stripe")


def test_detect_ats_from_url_lever():
    t = detect_ats_from_url("https://jobs.lever.co/anthropic")
    assert t == ATSTarget(provider="lever", token="anthropic")


def test_detect_ats_from_url_workable():
    t = detect_ats_from_url("https://apply.workable.com/huggingface")
    assert t == ATSTarget(provider="workable", token="huggingface")


def test_detect_ats_from_html_embedded():
    html = '<iframe src="https://boards.greenhouse.io/openai/jobs"></iframe>'
    t = detect_ats_from_html(html)
    assert t is not None
    assert t.provider == "greenhouse"
    assert t.token == "openai"


# ---------------------------------------------------------------------------
# HTML link extraction
# ---------------------------------------------------------------------------

def test_extract_jobs_from_html_finds_job_links():
    html = """
    <html><body>
      <a href="/careers/data-scientist-remote">Data Scientist - Remote</a>
      <a href="/about">About Us</a>
      <a href="/careers/ml-engineer">Machine Learning Engineer</a>
    </body></html>
    """
    jobs = extract_jobs_from_html(html, "Acme Corp", "https://acme.com/careers")
    titles = {j["title"] for j in jobs}
    assert "Data Scientist - Remote" in titles
    assert "Machine Learning Engineer" in titles
    assert all(j["company"] == "Acme Corp" for j in jobs)


# ---------------------------------------------------------------------------
# Role filtering
# ---------------------------------------------------------------------------

def test_filter_by_roles_keeps_matching_titles():
    jobs = [
        {"title": "Senior Data Scientist"},
        {"title": "Account Executive"},
        {"title": "ML Engineer, Infrastructure"},
    ]
    filtered = _filter_by_roles(jobs, ["Data Scientist", "Machine Learning Engineer"])
    titles = [j["title"] for j in filtered]
    assert "Senior Data Scientist" in titles
    assert "ML Engineer, Infrastructure" in titles
    assert "Account Executive" not in titles


def test_filter_by_roles_empty_roles_returns_all():
    jobs = [{"title": "Anything"}]
    assert _filter_by_roles(jobs, []) == jobs


# ---------------------------------------------------------------------------
# ATS fetch (mocked HTTP)
# ---------------------------------------------------------------------------

def test_fetch_greenhouse_jobs():
    payload = {
        "jobs": [{
            "id": 1,
            "title": "Data Scientist",
            "absolute_url": "https://boards.greenhouse.io/co/jobs/1",
            "location": {"name": "Remote"},
            "content": "Build models",
        }]
    }
    with patch("jobpilot.ats_fetchers._http_get_json", return_value=payload):
        jobs = fetch_ats_jobs(ATSTarget("greenhouse", "co"), "Co")
    assert len(jobs) == 1
    assert jobs[0]["title"] == "Data Scientist"
    assert jobs[0]["source"] == "greenhouse"


# ---------------------------------------------------------------------------
# CompanyScraper CSV loading
# ---------------------------------------------------------------------------

def test_load_companies_from_csv(tmp_path):
    csv_path = tmp_path / "companies.csv"
    csv_path.write_text(
        "Company,Career Page\n"
        "Stripe,https://stripe.com/jobs\n"
        "No URL Co,\n",
        encoding="utf-8",
    )
    scraper = CompanyScraper(companies_csv=csv_path, cursor_file=tmp_path / "cursor.json")
    companies = scraper._load_companies()
    assert len(companies) == 2
    assert companies[0]["company"] == "Stripe"
    assert companies[0]["url"] == "https://stripe.com/jobs"
    assert companies[1]["url"] == ""


def test_cursor_status_includes_url_count(tmp_path):
    csv_path = tmp_path / "companies.csv"
    csv_path.write_text(
        "Company,Career Page\nA,https://a.com\nB,\n",
        encoding="utf-8",
    )
    scraper = CompanyScraper(companies_csv=csv_path, cursor_file=tmp_path / "cursor.json")
    status = scraper.cursor_status()
    assert status["total_companies"] == 2
    assert status["companies_with_url"] == 1
    assert status["scraper_mode"] == "playwright+ats"
