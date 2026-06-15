from __future__ import annotations

"""JobPilot pipeline orchestrator + APScheduler 5 AM cron.

Pipeline order (runs each morning):
    0. Scrape career pages from your curated company list (Playwright + ATS APIs)
    1. Load resume text from data/master_resume.txt (or PDF)
    2. Score any unscored jobs already in the DB
    3. For top matches (>=75): tailor resume + generate cover letter + compile PDFs
    4. Generate morning report
    5. Send email digest

AWS EventBridge triggers this container at 5 AM Pacific via ECS task.
APScheduler is the in-process fallback for local dev / non-ECS deploys.

[INTERVIEW GOLD] Two scheduling layers: APScheduler handles in-process cron;
EventBridge + ECS provides reliable cloud execution with no always-on server.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger("jobpilot.scheduler")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    db_url:         Optional[str] = None,
    resume_path:    Optional[str] = None,
    outputs_root:   str           = "outputs/jobs",
    send_email:     bool          = False,
    top_n:          int           = 5,
    search_jobs:    bool          = True,
    search_batch:   int           = 30,
    progress_cb=None,
) -> dict:
    """Run the full morning pipeline. Returns a summary dict."""
    db_url = db_url or os.getenv("DATABASE_URL", "sqlite:///jobpilot_w5.db")
    start  = datetime.utcnow()
    log.info("Pipeline starting at %s", start.isoformat())

    # 0. Search company list for fresh job openings
    searched_count = 0
    if search_jobs:
        searched_count = _search_company_jobs(db_url, search_batch, progress_cb)
        log.info("Ingested %d new jobs from company search", searched_count)

    # 1. Load resume text
    resume_text = _load_resume(resume_path)

    # 2. Score any new (unscored) jobs in DB
    scored_count = _score_new_jobs(resume_text, db_url)
    log.info("Scored %d new jobs", scored_count)

    # 3. Tailor materials for top matches
    tailored_count = _tailor_top_matches(resume_text, db_url, outputs_root, top_n)
    log.info("Tailored materials for %d jobs", tailored_count)

    # 4. Generate report
    from jobpilot.agents.report_agent import ReportGenerator
    report = ReportGenerator(db_url=db_url, outputs_root=outputs_root).generate(top_n=top_n)
    log.info("Report generated: %d top matches", report.top_matches_count)

    # 5. Email (optional)
    email_sent = False
    if send_email:
        try:
            from jobpilot.agents.report_agent import EmailSender
            EmailSender().send(report)
            email_sent = True
            log.info("Email sent")
        except Exception as exc:
            log.warning("Email failed: %s", exc)

    elapsed = (datetime.utcnow() - start).total_seconds()
    summary = {
        "run_at":        start.isoformat(),
        "elapsed_s":     round(elapsed, 1),
        "searched":      searched_count,
        "scored":        scored_count,
        "tailored":      tailored_count,
        "top_matches":   report.top_matches_count,
        "email_sent":    email_sent,
        "report_date":   report.report_date,
    }
    log.info("Pipeline complete: %s", summary)
    return summary


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def _search_company_jobs(db_url: str, batch_size: int, progress_cb=None) -> int:
    """Step 0: scrape career pages from the curated company list."""
    try:
        from jobpilot.company_scraper import scrape_company_jobs
        from jobpilot.db_repository import JobRepository

        jobs = scrape_company_jobs(batch_size=batch_size, progress_cb=progress_cb)
        if not jobs:
            return 0

        repo  = JobRepository(db_url=db_url)
        count = 0
        for job in jobs:
            try:
                repo.add_job(job)
                count += 1
            except Exception:
                pass   # duplicate — already in DB
        return count
    except Exception as exc:
        log.warning("Company job search failed: %s", exc)
        return 0


def _load_resume(resume_path: Optional[str]) -> str:
    candidates = [
        resume_path,
        os.getenv("RESUME_PATH"),
        "data/master_resume.txt",
        "data/master_resume.pdf",
    ]
    for path in candidates:
        if not path:
            continue
        p = Path(path)
        if not p.exists():
            continue
        if p.suffix == ".pdf":
            try:
                from jobpilot.etl import ResumeETL
                return ResumeETL().extract_text_from_pdf(str(p))
            except Exception:
                continue
        return p.read_text(encoding="utf-8")
    log.warning("No resume file found — scoring will use empty resume text")
    return ""


def _score_new_jobs(resume_text: str, db_url: str) -> int:
    """Score any JobListings that have no entry in job_scores."""
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker
    from jobpilot.db_models import JobListing, JobScore, init_db
    from jobpilot.scoring_agent import ScoringAgent
    from jobpilot.db_repository import JobRepository

    SessionLocal = init_db(db_url)
    repo         = JobRepository(db_url)
    agent        = ScoringAgent()

    with SessionLocal() as session:
        # Jobs with no score row
        scored_ids = {row.job_id for row in session.query(JobScore.job_id).all()}
        unscored   = (
            session.query(JobListing)
            .filter(~JobListing.job_id.in_(scored_ids))
            .limit(50)
            .all()
        )

    count = 0
    for job_row in unscored:
        job = {
            "job_id":      job_row.job_id,
            "title":       job_row.title,
            "company":     job_row.company,
            "description": job_row.description,
            "skills":      [],
        }
        try:
            result = agent.score(resume_text, job)
            repo.add_score({
                "job_id":       job_row.job_id,
                "match_score":  result.match_score,
                "similarity":   0.0,
                "skill_gaps":   json_dumps(result.skill_gaps),
                "role_fit":     result.role_fit,
                "company_tier": result.company_tier,
            })
            count += 1
        except Exception as exc:
            log.warning("Scoring failed for %s: %s", job_row.job_id, exc)

    return count


def _tailor_top_matches(
    resume_text: str,
    db_url: str,
    outputs_root: str,
    top_n: int,
) -> int:
    """Generate tailored resume + cover letter PDFs for top matches."""
    import shutil
    from jobpilot.db_repository import JobRepository
    from jobpilot.resume_tailor import LLMResumeTailor, ResumeTailor
    from jobpilot.cover_letter import LLMCoverLetterGenerator, build_request
    from jobpilot.pdf_generator import PDFGenerator

    template_path = Path("data/resume_template.tex")
    if not template_path.exists():
        log.warning("resume_template.tex not found — skipping PDF generation")
        return 0

    # Check pdflatex available
    if not shutil.which("pdflatex"):
        log.warning("pdflatex not on PATH — skipping PDF generation")
        return 0

    template_tex = template_path.read_text(encoding="utf-8")
    repo         = JobRepository(db_url)
    min_score    = int(os.getenv("JOBPILOT_MIN_MATCH_SCORE", "40"))
    top_jobs     = repo.get_top_jobs_with_scores(limit=top_n, min_score=min_score)

    tailor  = LLMResumeTailor()
    cl_gen  = LLMCoverLetterGenerator()
    gen     = PDFGenerator(outputs_root=outputs_root)

    count = 0
    for row in top_jobs:
        job_id  = row["job_id"]
        out_dir = Path(outputs_root) / job_id

        # Skip if already generated
        if (out_dir / "tailored_resume.pdf").exists():
            continue

        try:
            tailored_tex = tailor.tailor_latex(template_tex, row)
            bullets      = ResumeTailor().tailor(resume_text, row).bullets if resume_text else []
            candidate_name      = os.getenv("CANDIDATE_NAME", "Candidate")
            candidate_email     = os.getenv("CANDIDATE_EMAIL", "")
            candidate_phone     = os.getenv("CANDIDATE_PHONE", "")
            candidate_linkedin  = os.getenv("CANDIDATE_LINKEDIN", "")
            candidate_portfolio = os.getenv("CANDIDATE_PORTFOLIO", "")
            cl_req  = build_request(
                candidate_name, row, bullets,
                candidate_email=candidate_email,
                candidate_phone=candidate_phone,
                candidate_linkedin=candidate_linkedin,
                candidate_portfolio=candidate_portfolio,
            )
            cl_text = cl_gen.generate(cl_req)
            gen.resume_pdf(tailored_tex, job_id)
            gen.cover_letter_pdf(
                cl_text, row, job_id,
                candidate_name=candidate_name,
                candidate_email=candidate_email,
                candidate_phone=candidate_phone,
                candidate_linkedin=candidate_linkedin,
                candidate_portfolio=candidate_portfolio,
            )
            count += 1
        except Exception as exc:
            log.warning("Tailoring failed for %s: %s", job_id, exc)

    return count


# ---------------------------------------------------------------------------
# APScheduler setup
# ---------------------------------------------------------------------------

def start_scheduler(
    hour:         int  = 5,
    minute:       int  = 0,
    db_url:       Optional[str] = None,
    send_email:   bool = True,
    run_now_also: bool = False,
) -> None:
    """Start APScheduler with a daily cron at `hour:minute` local time.

    Blocks until interrupted. For AWS ECS, use EventBridge instead — call
    run_pipeline() directly from the container entrypoint.
    """
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    scheduler = BlockingScheduler(timezone="America/Los_Angeles")
    scheduler.add_job(
        run_pipeline,
        CronTrigger(hour=hour, minute=minute),
        kwargs={"db_url": db_url, "send_email": send_email},
        id="morning_pipeline",
        name=f"JobPilot morning pipeline ({hour:02d}:{minute:02d} PT)",
        replace_existing=True,
    )

    if run_now_also:
        log.info("Running pipeline immediately before scheduling")
        run_pipeline(db_url=db_url, send_email=send_email)

    log.info("Scheduler started — next run at %02d:%02d PT", hour, minute)
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Scheduler stopped")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def json_dumps(obj) -> str:
    import json
    try:
        return json.dumps(obj)
    except Exception:
        return "[]"


# ---------------------------------------------------------------------------
# Entrypoint (python -m jobpilot.scheduler.cron)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="JobPilot scheduler")
    parser.add_argument("--run-now", action="store_true", help="Run pipeline immediately then schedule")
    parser.add_argument("--no-email", action="store_true", help="Skip email delivery")
    parser.add_argument("--hour",   type=int, default=5)
    parser.add_argument("--minute", type=int, default=0)
    args = parser.parse_args()

    start_scheduler(
        hour=args.hour,
        minute=args.minute,
        send_email=not args.no_email,
        run_now_also=args.run_now,
    )
