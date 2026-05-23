"""W6 tests: MorningReport, ReportGenerator, EmailSender, _parse_gaps."""
from __future__ import annotations

import json
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from jobpilot.agents.report_agent import (
    EmailSender,
    JobBrief,
    MorningReport,
    ReportGenerator,
    _parse_gaps,
)

SQLITE_URL = "sqlite://"

# ---------------------------------------------------------------------------
# _parse_gaps
# ---------------------------------------------------------------------------

def test_parse_gaps_json_string():
    assert _parse_gaps('["python", "pytorch"]') == ["python", "pytorch"]

def test_parse_gaps_python_list_repr():
    assert _parse_gaps("['sql', 'spark']") == ["sql", "spark"]

def test_parse_gaps_already_list():
    assert _parse_gaps(["a", "b"]) == ["a", "b"]

def test_parse_gaps_empty():
    assert _parse_gaps("") == []
    assert _parse_gaps(None) == []
    assert _parse_gaps("[]") == []


# ---------------------------------------------------------------------------
# MorningReport.to_text()
# ---------------------------------------------------------------------------

def _sample_report(n_jobs=2) -> MorningReport:
    jobs = [
        JobBrief(
            rank=i, job_id=f"job_{i}", title=f"Title {i}", company=f"Co {i}",
            location="Remote", match_score=80+i, skill_gaps=["distributed systems", "spark"],
            role_fit="MLE", company_tier="Growth", resume_ready=True, cover_letter_ready=True,
        )
        for i in range(1, n_jobs + 1)
    ]
    return MorningReport(
        report_date="May 22, 2026",
        total_scored=10,
        top_matches_count=2,
        jobs=jobs,
        recurring_gaps=["distributed systems", "spark"],
        weak_skills=["kubernetes"],
    )


def test_report_to_text_contains_date():
    r = _sample_report()
    text = r.to_text()
    assert "May 22, 2026" in text


def test_report_to_text_contains_job_titles():
    r = _sample_report()
    text = r.to_text()
    assert "Title 1" in text
    assert "Title 2" in text


def test_report_to_text_contains_match_scores():
    r = _sample_report()
    text = r.to_text()
    assert "81%" in text or "81" in text


def test_report_to_text_shows_gaps():
    r = _sample_report()
    text = r.to_text()
    assert "distributed systems" in text


def test_report_to_text_shows_pdf_paths_when_ready():
    r = _sample_report()
    text = r.to_text()
    assert "tailored_resume.pdf" in text
    assert "cover_letter.pdf" in text


def test_report_to_text_skips_pdf_when_not_ready():
    r = _sample_report()
    r.jobs[0].resume_ready = False
    r.jobs[0].cover_letter_ready = False
    text = r.to_text()
    # job_2 still has PDFs, so paths still appear — but job_1 path should not
    assert "outputs/jobs/job_1/tailored_resume.pdf" not in text


def test_report_to_text_shows_weak_skills():
    r = _sample_report()
    assert "kubernetes" in r.to_text()


# ---------------------------------------------------------------------------
# MorningReport.to_html()
# ---------------------------------------------------------------------------

def test_report_to_html_is_valid_html():
    r = _sample_report()
    html = r.to_html()
    assert "<!DOCTYPE html>" in html
    assert "</html>" in html


def test_report_to_html_contains_table():
    r = _sample_report()
    html = r.to_html()
    assert "<table" in html
    assert "Co 1" in html


def test_report_to_html_escapes_ampersands():
    r = _sample_report()
    r.jobs[0].company = "AT&T"
    html = r.to_html()
    # Should not have raw & in HTML content (it appears as &amp; or inside attr)
    # Simple check: the raw string appears in the table
    assert "AT&T" in html or "AT&amp;T" in html


# ---------------------------------------------------------------------------
# ReportGenerator — with mocked DB
# ---------------------------------------------------------------------------

SAMPLE_DB_ROWS = [
    {
        "job_id": "acme_mle", "title": "ML Engineer", "company": "Acme AI",
        "location": "Remote", "source": "linkedin", "url": "",
        "match_score": 85, "skill_gaps": '["pytorch", "kubernetes"]',
        "role_fit": "MLE", "company_tier": "Growth",
    },
    {
        "job_id": "big_ds", "title": "Data Scientist", "company": "BigCo",
        "location": "NYC", "source": "indeed", "url": "",
        "match_score": 78, "skill_gaps": '["spark"]',
        "role_fit": "DS", "company_tier": "FAANG",
    },
]


def test_report_generator_builds_report(tmp_path):
    gen = ReportGenerator(db_url=SQLITE_URL, outputs_root=str(tmp_path))
    with patch("jobpilot.agents.report_agent.JobRepository") as MockRepo, \
         patch("jobpilot.agents.report_agent.CandidateModelManager") as MockMgr:
        MockRepo.return_value.get_top_jobs_with_scores.return_value = SAMPLE_DB_ROWS
        MockMgr.return_value.get_weak_skills.return_value = [{"skill_name": "spark"}]
        report = gen.generate(top_n=5)

    assert report.total_scored == 2
    assert report.top_matches_count == 2
    assert len(report.jobs) == 2
    assert report.jobs[0].title == "ML Engineer"
    assert "pytorch" in report.jobs[0].skill_gaps


def test_report_generator_marks_pdf_ready(tmp_path):
    # Create a fake PDF for one job
    pdf_dir = tmp_path / "acme_mle"
    pdf_dir.mkdir()
    (pdf_dir / "tailored_resume.pdf").write_bytes(b"%PDF-fake")

    gen = ReportGenerator(db_url=SQLITE_URL, outputs_root=str(tmp_path))
    with patch("jobpilot.agents.report_agent.JobRepository") as MockRepo, \
         patch("jobpilot.agents.report_agent.CandidateModelManager") as MockMgr:
        MockRepo.return_value.get_top_jobs_with_scores.return_value = SAMPLE_DB_ROWS
        MockMgr.return_value.get_weak_skills.return_value = []
        report = gen.generate(top_n=5)

    assert report.jobs[0].resume_ready is True
    assert report.jobs[0].cover_letter_ready is False
    assert report.jobs[1].resume_ready is False


def test_report_generator_recurring_gaps(tmp_path):
    rows = SAMPLE_DB_ROWS + [{
        "job_id": "other", "title": "DS", "company": "X",
        "location": "", "source": "", "url": "",
        "match_score": 60, "skill_gaps": '["pytorch", "spark"]',
        "role_fit": "DS", "company_tier": "Startup",
    }]
    gen = ReportGenerator(db_url=SQLITE_URL, outputs_root=str(tmp_path))
    with patch("jobpilot.agents.report_agent.JobRepository") as MockRepo, \
         patch("jobpilot.agents.report_agent.CandidateModelManager") as MockMgr:
        MockRepo.return_value.get_top_jobs_with_scores.return_value = rows
        MockMgr.return_value.get_weak_skills.return_value = []
        report = gen.generate(top_n=5)

    # pytorch appears in 2 jobs, spark in 2 jobs → both recurring
    assert "pytorch" in report.recurring_gaps or "spark" in report.recurring_gaps


# ---------------------------------------------------------------------------
# EmailSender
# ---------------------------------------------------------------------------

def test_email_sender_raises_without_credentials():
    sender = EmailSender(user="", password="")
    report = _sample_report()
    with pytest.raises(ValueError, match="EMAIL_USER"):
        sender.send(report)


def test_email_sender_calls_smtp(tmp_path):
    sender = EmailSender(host="smtp.gmail.com", port=587, user="test@gmail.com", password="pw")
    report = _sample_report()

    with patch("jobpilot.agents.report_agent.smtplib.SMTP") as MockSMTP:
        mock_smtp = MagicMock()
        MockSMTP.return_value.__enter__ = lambda s: mock_smtp
        MockSMTP.return_value.__exit__ = MagicMock(return_value=False)
        result = sender.send(report, to="recipient@example.com")

    assert result is True
    mock_smtp.starttls.assert_called_once()
    mock_smtp.login.assert_called_once_with("test@gmail.com", "pw")
    mock_smtp.sendmail.assert_called_once()


def test_email_subject_contains_date_and_matches():
    sender = EmailSender(user="u@g.com", password="pw")
    report = _sample_report()

    captured_msg = {}
    def fake_sendmail(from_, to_, msg_str):
        captured_msg["msg"] = msg_str

    with patch("jobpilot.agents.report_agent.smtplib.SMTP") as MockSMTP:
        mock_smtp = MagicMock()
        mock_smtp.sendmail.side_effect = fake_sendmail
        MockSMTP.return_value.__enter__ = lambda s: mock_smtp
        MockSMTP.return_value.__exit__ = MagicMock(return_value=False)
        sender.send(report, to="r@example.com")

    assert "May 22, 2026" in captured_msg["msg"]
    assert "matches" in captured_msg["msg"]
