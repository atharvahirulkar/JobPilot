from __future__ import annotations

import ast
import json
import os
import smtplib
import textwrap
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Optional

from jobpilot.db_repository import JobRepository
from jobpilot.candidate_model.model import CandidateModelManager


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class JobBrief:
    rank:               int
    job_id:             str
    title:              str
    company:            str
    location:           str
    match_score:        int
    skill_gaps:         List[str]
    role_fit:           str
    company_tier:       str
    resume_ready:       bool
    cover_letter_ready: bool
    url:                str = ""


@dataclass
class MorningReport:
    report_date:        str
    total_scored:       int
    top_matches_count:  int
    jobs:               List[JobBrief]
    recurring_gaps:     List[str]
    weak_skills:        List[str] = field(default_factory=list)

    def to_text(self) -> str:
        lines = [
            f"Good morning. Here's your {self.report_date} job brief.",
            "",
            f"Scored {self.total_scored} jobs  |  {self.top_matches_count} strong matches (score >= 75)",
            "",
        ]
        for brief in self.jobs:
            gaps = ", ".join(brief.skill_gaps[:3]) or "none identified"
            lines += [
                f"#{brief.rank}  {brief.title} — {brief.company}  [{brief.company_tier}]",
                f"    Match: {brief.match_score}%  |  Fit: {brief.role_fit}  |  Gaps: {gaps}",
            ]
            if brief.resume_ready:
                lines.append(f"    Resume  → outputs/jobs/{brief.job_id}/tailored_resume.pdf")
            if brief.cover_letter_ready:
                lines.append(f"    Cover   → outputs/jobs/{brief.job_id}/cover_letter.pdf")
            lines.append("")

        if self.recurring_gaps:
            lines.append("Recurring skill gaps across all JDs:")
            for gap in self.recurring_gaps[:5]:
                lines.append(f"  • {gap}")
            lines.append("")

        if self.weak_skills:
            lines.append("Interview focus areas (weakest skills):")
            for skill in self.weak_skills[:3]:
                lines.append(f"  • {skill}")
            lines.append("")

        lines.append("Open dashboard → http://localhost:8501")
        return "\n".join(lines)

    def to_html(self) -> str:
        rows = ""
        for b in self.jobs:
            gaps_html = ", ".join(b.skill_gaps[:3]) or "—"
            mat_html  = "✓" if b.resume_ready else "—"
            cl_html   = "✓" if b.cover_letter_ready else "—"
            rows += (
                f"<tr>"
                f"<td style='padding:6px 10px'><b>{b.rank}</b></td>"
                f"<td style='padding:6px 10px'>{b.title}</td>"
                f"<td style='padding:6px 10px'>{b.company}</td>"
                f"<td style='padding:6px 10px;text-align:center'>{b.match_score}%</td>"
                f"<td style='padding:6px 10px'>{b.role_fit}</td>"
                f"<td style='padding:6px 10px'>{b.company_tier}</td>"
                f"<td style='padding:6px 10px'>{gaps_html}</td>"
                f"<td style='padding:6px 10px;text-align:center'>{mat_html}</td>"
                f"<td style='padding:6px 10px;text-align:center'>{cl_html}</td>"
                f"</tr>\n"
            )

        gaps_list = "".join(f"<li>{g}</li>" for g in self.recurring_gaps[:5])
        weak_list = "".join(f"<li>{s}</li>" for s in self.weak_skills[:3])

        return textwrap.dedent(f"""
        <!DOCTYPE html>
        <html>
        <head><meta charset="utf-8"></head>
        <body style="font-family:Arial,sans-serif;max-width:800px;margin:0 auto;padding:20px;color:#222">
          <h2 style="border-bottom:2px solid #333;padding-bottom:8px">
            JobPilot — {self.report_date}
          </h2>
          <p style="font-size:15px">
            Scored <b>{self.total_scored}</b> jobs &nbsp;|&nbsp;
            <b style="color:#1a7a1a">{self.top_matches_count} strong matches</b> (≥75%)
          </p>

          <h3>Top Matches</h3>
          <table style="border-collapse:collapse;width:100%;font-size:13px">
            <thead style="background:#f0f0f0">
              <tr>
                <th style="padding:6px 10px">#</th>
                <th style="padding:6px 10px;text-align:left">Title</th>
                <th style="padding:6px 10px;text-align:left">Company</th>
                <th style="padding:6px 10px">Score</th>
                <th style="padding:6px 10px;text-align:left">Fit</th>
                <th style="padding:6px 10px;text-align:left">Tier</th>
                <th style="padding:6px 10px;text-align:left">Gaps</th>
                <th style="padding:6px 10px">Resume</th>
                <th style="padding:6px 10px">CL</th>
              </tr>
            </thead>
            <tbody>{rows}</tbody>
          </table>

          {"<h3>Recurring Skill Gaps</h3><ul>" + gaps_list + "</ul>" if gaps_list else ""}
          {"<h3>Interview Focus (Weakest Skills)</h3><ul>" + weak_list + "</ul>" if weak_list else ""}

          <p style="margin-top:24px;font-size:12px;color:#888">
            JobPilot · <a href="http://localhost:8501">Open Dashboard</a>
          </p>
        </body>
        </html>
        """).strip()


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------

class ReportGenerator:
    """Build a MorningReport from the PostgreSQL jobs + scores tables.

    [INTERVIEW GOLD] Decoupled from delivery — same report object feeds
    both the email sender and the Streamlit dashboard (W7).
    """

    def __init__(self, db_url: Optional[str] = None, outputs_root: str = "outputs/jobs"):
        self.db_url       = db_url or os.getenv("DATABASE_URL", "sqlite:///jobpilot_w5.db")
        self.outputs_root = Path(outputs_root)

    def generate(self, top_n: int = 5, min_score: int = 0) -> MorningReport:
        repo = JobRepository(self.db_url)
        rows = repo.get_top_jobs_with_scores(limit=top_n * 3, min_score=min_score)

        # Count totals
        total_scored      = len(rows)
        top_matches_count = sum(1 for r in rows if r["match_score"] >= 75)

        # Build JobBrief list (top_n only)
        briefs: List[JobBrief] = []
        all_gaps: List[str]    = []

        for i, row in enumerate(rows[:top_n], start=1):
            gaps = _parse_gaps(row.get("skill_gaps", "[]"))
            all_gaps.extend(gaps)

            job_id = row["job_id"]
            resume_ready = (self.outputs_root / job_id / "tailored_resume.pdf").exists()
            cl_ready     = (self.outputs_root / job_id / "cover_letter.pdf").exists()

            briefs.append(JobBrief(
                rank=i,
                job_id=job_id,
                title=row["title"],
                company=row["company"],
                location=row.get("location", ""),
                match_score=row["match_score"],
                skill_gaps=gaps,
                role_fit=row.get("role_fit", ""),
                company_tier=row.get("company_tier", ""),
                resume_ready=resume_ready,
                cover_letter_ready=cl_ready,
                url=row.get("url", ""),
            ))

        # Top recurring gaps (most frequent across all scored jobs)
        gap_counts    = Counter(all_gaps)
        recurring     = [g for g, _ in gap_counts.most_common(5)]

        # Weak skills from candidate model
        try:
            mgr        = CandidateModelManager(db_url=self.db_url)
            weak_skills = [s["skill_name"] for s in mgr.get_weak_skills(limit=3)]
        except Exception:
            weak_skills = []

        return MorningReport(
            report_date=date.today().strftime("%B %-d, %Y"),
            total_scored=total_scored,
            top_matches_count=top_matches_count,
            jobs=briefs,
            recurring_gaps=recurring,
            weak_skills=weak_skills,
        )


# ---------------------------------------------------------------------------
# Email sender
# ---------------------------------------------------------------------------

class EmailSender:
    """Send the morning report via SMTP (Gmail by default).

    Configure via env vars:
        EMAIL_HOST      smtp.gmail.com
        EMAIL_PORT      587
        EMAIL_USER      your@gmail.com
        EMAIL_PASSWORD  app-password (not your Google password)
        REPORT_TO       recipient@example.com  (defaults to EMAIL_USER)
    """

    def __init__(
        self,
        host:     Optional[str] = None,
        port:     Optional[int] = None,
        user:     Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.host     = host     or os.getenv("EMAIL_HOST",     "smtp.gmail.com")
        self.port     = port     or int(os.getenv("EMAIL_PORT", "587"))
        self.user     = user     or os.getenv("EMAIL_USER",     "")
        self.password = password or os.getenv("EMAIL_PASSWORD", "")

    def send(self, report: MorningReport, to: Optional[str] = None) -> bool:
        """Send the report HTML email. Returns True on success."""
        if not self.user or not self.password:
            raise ValueError("EMAIL_USER and EMAIL_PASSWORD must be set")

        recipient = to or os.getenv("REPORT_TO", self.user)
        subject   = f"JobPilot - {report.report_date} - {report.top_matches_count} matches"

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = self.user
        msg["To"]      = recipient
        msg.attach(MIMEText(report.to_text(), "plain"))
        msg.attach(MIMEText(report.to_html(), "html"))

        with smtplib.SMTP(self.host, self.port) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(self.user, self.password)
            smtp.sendmail(self.user, recipient, msg.as_string())

        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_gaps(raw: Any) -> List[str]:
    """Parse skill_gaps which may be a JSON string, Python list repr, or list."""
    if isinstance(raw, list):
        return [str(g) for g in raw]
    if not raw:
        return []
    try:
        parsed = json.loads(str(raw))
        return [str(g) for g in parsed] if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        parsed = ast.literal_eval(str(raw))
        return [str(g) for g in parsed] if isinstance(parsed, list) else []
    except Exception:
        return []
