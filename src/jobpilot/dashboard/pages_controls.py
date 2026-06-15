"""Controls page — search jobs, run pipeline, manage the morning scheduler.

Everything that previously required a terminal command lives here.
"""

import os
import json
import tempfile
import traceback
from pathlib import Path

import streamlit as st


def _get_db_url():
    return st.session_state.get("db_url") or os.getenv("DATABASE_URL", "sqlite:///jobpilot_w5.db")


def render():
    st.header("Controls")
    st.markdown(
        "Search for jobs, run the full pipeline, and manage the scheduler — "
        "no terminal needed."
    )

    db_url       = _get_db_url()
    outputs_root = st.session_state.get("outputs_root", "outputs/jobs")

    # ── 0. Search Company List ─────────────────────────────────────────────
    st.subheader("0. Search Your Target Companies")
    _render_company_search(db_url)

    st.markdown("---")

    # ── 1. Run Full Pipeline ───────────────────────────────────────────────
    st.subheader("1. Run Full Pipeline")
    _render_pipeline(db_url, outputs_root)

    st.markdown("---")

    # ── 2. Score a Single JD ──────────────────────────────────────────────
    st.subheader("2. Score a Single Job (Quick)")
    _render_quick_score()

    st.markdown("---")

    # ── 3. Manual Job Import ──────────────────────────────────────────────
    st.subheader("3. Manual Job Import")
    _render_manual_import(db_url)

    st.markdown("---")

    # ── 4. Scheduler ──────────────────────────────────────────────────────
    st.subheader("4. Morning Scheduler")
    _render_scheduler()


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def _render_company_search(db_url: str):
    """Section 0 — scrape career pages from the curated company list."""

    try:
        from jobpilot.company_scraper import CompanyScraper
        scraper = CompanyScraper()
        status  = scraper.cursor_status()
    except Exception as e:
        st.error(f"Could not load scraper: {e}")
        return

    # Status card
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Target Companies",      status["total_companies"])
    col2.metric("With Career Page URL",  status.get("companies_with_url", 0))
    col3.metric("Days for Full Cycle",   status["days_to_full_cycle"])
    col4.metric("Last Run",              status["last_date"] or "Never")

    if status["next_batch_preview"]:
        st.caption(
            "Next batch preview: "
            + ", ".join(status["next_batch_preview"])
            + ("…" if status["total_companies"] > 5 else "")
        )

    st.caption(
        "Scrapes each company's **Career Page** URL (Playwright + ATS APIs). "
        "No SerpAPI credits. Optional OpenAI parsing for custom pages."
    )

    col_left, col_right = st.columns([3, 1])
    with col_left:
        batch_size = st.slider(
            "Companies to scrape this run",
            min_value=5, max_value=100, value=30,
            help="One page fetch per company. ~30 companies ≈ 2–5 min.",
        )
        roles_input = st.multiselect(
            "Roles to keep (title filter)",
            options=[
                "Data Scientist",
                "Machine Learning Engineer",
                "Data Engineer",
                "Software Engineer",
                "AI Engineer",
                "Quantitative Researcher",
                "Quantitative Analyst",
                "Research Scientist",
                "Applied Scientist",
                "Analytics Engineer",
                "MLOps Engineer",
                "NLP Engineer",
                "Computer Vision Engineer",
                "Business Intelligence Engineer",
            ],
            default=[
                "Data Scientist",
                "Machine Learning Engineer",
                "Data Engineer",
                "Software Engineer",
                "AI Engineer",
                "Quantitative Researcher",
            ],
            help="Filters job titles after scraping — no extra API calls.",
        )
        use_llm = st.checkbox(
            "Use OpenAI fallback for custom career pages",
            value=bool(os.getenv("OPENAI_API_KEY")),
            help="Only used when ATS detection and link extraction find nothing.",
        )
    with col_right:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        search_btn = st.button(
            "🔍 Scrape Career Pages", type="primary", use_container_width=True
        )

    if search_btn:
        progress_bar  = st.progress(0.0, text="Starting scrape…")
        status_text   = st.empty()

        def update_progress(msg: str, current: int, total: int):
            frac = current / total if total else 0
            progress_bar.progress(frac, text=msg)
            status_text.caption(f"{current}/{total} companies scraped")

        with st.spinner("Scraping career pages (Playwright)…"):
            try:
                from jobpilot.company_scraper import CompanyScraper
                from jobpilot.db_repository import JobRepository

                scraper = CompanyScraper(
                    batch_size=batch_size,
                    roles=roles_input,
                    use_llm_fallback=use_llm,
                )
                jobs = scraper.scrape(progress_cb=update_progress)

                progress_bar.empty()
                status_text.empty()

                repo     = JobRepository(db_url=db_url)
                imported = 0
                skipped  = 0
                for job in jobs:
                    try:
                        repo.add_job(job)
                        imported += 1
                    except Exception:
                        skipped += 1

                c1, c2, c3 = st.columns(3)
                c1.metric("Jobs Found",          len(jobs))
                c2.metric("New (added to DB)",   imported)
                c3.metric("Already Known (dup)", skipped)

                if imported:
                    st.success(
                        f"Found {imported} new openings! "
                        "Go to **Run Full Pipeline** below to score them against your resume."
                    )
                else:
                    st.info(
                        "No new jobs found in this batch — all results already in the database, "
                        "or no open roles at these companies right now."
                    )
            except Exception as e:
                progress_bar.empty()
                st.error(f"Search failed: {e}")
                st.code(traceback.format_exc())


def _render_pipeline(db_url: str, outputs_root: str):
    st.markdown(
        "Scores all unscored jobs against your resume (LLM), "
        "generates tailored resume + cover letter PDFs for top matches (≥ 75%), "
        "and produces the morning report."
    )

    col_left, col_right = st.columns([3, 1])
    with col_left:
        top_n        = st.slider("Top N jobs to generate materials for", 1, 20, 5)
        search_first = st.checkbox(
            "Also search company list before scoring",
            value=True,
            help="Runs the company search (step 0) as part of the pipeline.",
        )
        search_batch = st.slider(
            "Companies to search (if enabled)",
            5, 100, 30,
            disabled=not search_first,
        )
        send_email   = st.checkbox("Send email digest after run", value=False)
    with col_right:
        st.markdown("<br><br><br><br>", unsafe_allow_html=True)
        run_btn = st.button("▶ Run Pipeline", type="primary", use_container_width=True)

    if run_btn:
        pipeline_progress = st.progress(0.0, text="Starting…")
        pipeline_status   = st.empty()

        def pipeline_progress_cb(msg: str, current: int, total: int):
            frac = current / total if total else 0
            pipeline_progress.progress(min(frac * 0.4, 0.4), text=f"[Search] {msg}")
            pipeline_status.caption(msg)

        with st.spinner("Running full pipeline…"):
            try:
                from jobpilot.scheduler.cron import run_pipeline

                pipeline_progress.progress(0.1, text="Searching company boards…")
                summary = run_pipeline(
                    db_url=db_url,
                    outputs_root=outputs_root,
                    send_email=send_email,
                    top_n=top_n,
                    search_jobs=search_first,
                    search_batch=search_batch,
                    progress_cb=pipeline_progress_cb if search_first else None,
                )
                pipeline_progress.progress(1.0, text="Done!")
                pipeline_status.empty()

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Searched",           summary.get("searched", 0))
                c2.metric("Scored",             summary["scored"])
                c3.metric("Tailored",           summary["tailored"])
                c4.metric("Top Matches",        summary["top_matches"])
                c5.metric("Elapsed (s)",        summary["elapsed_s"])

                if summary.get("email_sent"):
                    st.success("Email digest sent.")
                st.success(
                    f"Pipeline complete — report date: {summary['report_date']}. "
                    "Switch to **Job Tracker** or **Morning Report** to see results."
                )
            except Exception as e:
                pipeline_progress.empty()
                st.error(f"Pipeline failed: {e}")
                st.code(traceback.format_exc())


def _render_quick_score():
    st.markdown("Paste any job description to get an instant match score against your resume.")

    jd_text    = st.text_area("Paste job description", height=160,
                              placeholder="Data Scientist @ Acme Corp\n\nWe are looking for…")
    jd_title   = st.text_input("Job title",   placeholder="Data Scientist")
    jd_company = st.text_input("Company",     placeholder="Acme Corp")

    if st.button("Score This JD", type="secondary"):
        if not jd_text.strip():
            st.warning("Paste a job description first.")
            return
        resume_text = _load_resume()
        if not resume_text:
            st.warning("No resume found at `data/master_resume.txt` / `.pdf`.")
            return
        with st.spinner("Scoring with LLM…"):
            try:
                from jobpilot.scoring_agent import ScoringAgent
                result = ScoringAgent().score(resume_text, {
                    "title":       jd_title or "Unknown",
                    "company":     jd_company or "Unknown",
                    "description": jd_text,
                    "skills":      [],
                })
                c1, c2, c3 = st.columns(3)
                c1.metric("Match Score",  f"{result.match_score}/100")
                c2.metric("Role Fit",     result.role_fit)
                c3.metric("Company Tier", result.company_tier)
                if result.skill_gaps:
                    st.markdown(f"**Skill Gaps:** {', '.join(result.skill_gaps)}")
                if result.is_top_match:
                    st.success("Top match (≥ 75%) — worth applying!")
                else:
                    st.info("Below 75% threshold.")
            except Exception as e:
                st.error(f"Scoring failed: {e}")
                st.code(traceback.format_exc())


def _render_manual_import(db_url: str):
    st.markdown(
        "Upload a JSON or CSV file of job listings to bulk-load into the database. "
        "Useful for importing from external sources or other job boards."
    )

    uploaded = st.file_uploader("Choose a jobs file", type=["json", "csv"],
                                help='JSON: list or {"jobs":[...]}. CSV: title, company, description columns.')
    if uploaded is None:
        return

    st.write(f"File: `{uploaded.name}` ({uploaded.size:,} bytes)")
    if st.button("Import", type="primary"):
        suffix = ".json" if uploaded.name.lower().endswith(".json") else ".csv"
        with st.spinner(f"Importing {uploaded.name}…"):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                    f.write(uploaded.getvalue())
                    tmp_path = f.name

                from jobpilot.job_imports import load_jobs_file
                from jobpilot.db_repository import JobRepository

                jobs     = load_jobs_file(tmp_path)
                Path(tmp_path).unlink(missing_ok=True)
                repo     = JobRepository(db_url=db_url)
                imported = skipped = 0
                for job in jobs:
                    try:
                        repo.add_job(job)
                        imported += 1
                    except Exception:
                        skipped += 1

                c1, c2, c3 = st.columns(3)
                c1.metric("Parsed",        len(jobs))
                c2.metric("Imported",      imported)
                c3.metric("Skipped (dup)", skipped)
                st.success(f"Imported {imported} jobs.")
            except Exception as e:
                st.error(f"Import failed: {e}")
                st.code(traceback.format_exc())


def _render_scheduler():
    col1, col2 = st.columns(2)
    with col1:
        sched_hour = st.number_input("Hour (24h local)", min_value=0, max_value=23, value=5)
    with col2:
        sched_min  = st.number_input("Minute", min_value=0, max_value=59, value=0)

    st.info(
        f"**To start the scheduler**, open a terminal once and run:\n\n"
        f"```\npython -m jobpilot.main schedule "
        f"--hour {int(sched_hour)} --minute {int(sched_min)}\n```\n\n"
        "It will search your company list, score all new jobs, tailor resumes for top matches, "
        "and generate your morning report — every day at this time, automatically."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_resume() -> str:
    for path in [os.getenv("RESUME_PATH", ""), "data/master_resume.txt", "data/master_resume.pdf"]:
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
    return ""
