"""Job Tracker page — ranked jobs with scores, skill gaps, and materials generation."""

import json
import os
from pathlib import Path

import streamlit as st


def _get_db_url():
    return st.session_state.get("db_url") or os.getenv("DATABASE_URL", "sqlite:///jobpilot_w5.db")


def _safe_parse_gaps(raw) -> list:
    """Parse skill gaps — accepts JSON list string or plain list. Never uses eval()."""
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def render():
    st.header("Job Tracker")
    st.markdown("Ranked jobs with match scores, skill gaps, and material generation.")

    db_url = _get_db_url()
    outputs_root = st.session_state.get("outputs_root", "outputs/jobs")

    try:
        from jobpilot.db_repository import JobRepository
        repo = JobRepository(db_url=db_url)
        jobs = repo.get_top_jobs_with_scores(limit=50, min_score=0)

        if not jobs:
            st.info(
                "No jobs in the database yet. "
                "Go to **Controls** to import a jobs file or run the pipeline."
            )
            return

        # Build display rows
        rows = []
        for i, j in enumerate(jobs):
            gaps = _safe_parse_gaps(j["skill_gaps"])
            rows.append({
                "Rank":    i + 1,
                "Title":   j["title"],
                "Company": j["company"],
                "Score":   f"{j['match_score']}%",
                "Fit":     j["role_fit"] or "-",
                "Tier":    j["company_tier"] or "-",
                "Gaps":    ", ".join(gaps[:2]) if gaps else "-",
            })

        st.dataframe(rows, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Job Details & Material Generation")

        selected_idx = st.number_input(
            "View details for job (rank number)",
            min_value=1,
            max_value=len(jobs),
            value=1,
            step=1,
        ) - 1
        job = jobs[selected_idx]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Match Score", f"{job['match_score']}/100")
        col2.metric("Role Fit",    job["role_fit"] or "-")
        col3.metric("Company Tier", job["company_tier"] or "-")
        col4.metric("Source",      job["source"])

        if job.get("url"):
            st.markdown(f"**URL:** [{job['url']}]({job['url']})")

        gaps = _safe_parse_gaps(job["skill_gaps"])
        if gaps:
            st.markdown(f"**Skill Gaps:** {', '.join(gaps)}")

        # Materials status
        job_id   = job["job_id"]
        out_dir  = Path(outputs_root) / job_id
        res_pdf  = out_dir / "tailored_resume.pdf"
        cl_pdf   = out_dir / "cover_letter.pdf"

        st.markdown("**Application Materials:**")
        col_r, col_c, col_gen = st.columns([2, 2, 1])

        with col_r:
            if res_pdf.exists():
                st.success(f"Resume ready: `{res_pdf}`")
            else:
                st.warning("Resume not yet generated")

        with col_c:
            if cl_pdf.exists():
                st.success(f"Cover letter ready: `{cl_pdf}`")
            else:
                st.warning("Cover letter not yet generated")

        with col_gen:
            if st.button("Generate Materials", type="primary", use_container_width=True):
                _generate_materials(job, db_url, outputs_root)

    except Exception as e:
        st.error(f"Error loading jobs: {e}")
        import traceback
        st.code(traceback.format_exc())


def _generate_materials(job: dict, db_url: str, outputs_root: str):
    """Tailor resume + generate cover letter PDFs for the selected job."""
    import shutil
    from pathlib import Path

    template_path = Path("data/resume_template.tex")
    if not template_path.exists():
        st.error("Resume template not found at `data/resume_template.tex`.")
        return

    if not shutil.which("pdflatex"):
        st.error("`pdflatex` not found on PATH. Install TeX Live / MikTeX to generate PDFs.")
        return

    candidate_name      = st.session_state.get("candidate_name")      or os.getenv("CANDIDATE_NAME", "")
    candidate_email     = st.session_state.get("candidate_email")     or os.getenv("CANDIDATE_EMAIL", "")
    candidate_phone     = st.session_state.get("candidate_phone")     or os.getenv("CANDIDATE_PHONE", "")
    candidate_linkedin  = st.session_state.get("candidate_linkedin")  or os.getenv("CANDIDATE_LINKEDIN", "")
    candidate_portfolio = st.session_state.get("candidate_portfolio") or os.getenv("CANDIDATE_PORTFOLIO", "")
    resume_path_env = os.getenv("RESUME_PATH", "data/master_resume.txt")

    resume_text = ""
    for rp in [resume_path_env, "data/master_resume.txt", "data/master_resume.pdf"]:
        p = Path(rp)
        if p.exists():
            if p.suffix == ".pdf":
                try:
                    from jobpilot.etl import ResumeETL
                    resume_text = ResumeETL().extract_text_from_pdf(str(p))
                except Exception:
                    pass
            else:
                resume_text = p.read_text(encoding="utf-8")
            break

    with st.spinner("Tailoring resume and generating cover letter…"):
        try:
            from jobpilot.resume_tailor import LLMResumeTailor, ResumeTailor
            from jobpilot.cover_letter import LLMCoverLetterGenerator, build_request
            from jobpilot.pdf_generator import PDFGenerator

            template_tex  = template_path.read_text(encoding="utf-8")
            tailored_tex  = LLMResumeTailor().tailor_latex(template_tex, job)

            bullets = ResumeTailor().tailor(resume_text, job).bullets if resume_text else []
            name    = candidate_name or "Candidate"
            cl_req  = build_request(
                name, job, bullets,
                candidate_email=candidate_email,
                candidate_phone=candidate_phone,
                candidate_linkedin=candidate_linkedin,
                candidate_portfolio=candidate_portfolio,
            )
            cl_text = LLMCoverLetterGenerator().generate(cl_req)

            gen = PDFGenerator(outputs_root=outputs_root)
            res_pdf = gen.resume_pdf(tailored_tex, job["job_id"])
            cl_pdf  = gen.cover_letter_pdf(
                cl_text, job, job["job_id"],
                candidate_name=name,
                candidate_email=candidate_email,
                candidate_phone=candidate_phone,
                candidate_linkedin=candidate_linkedin,
                candidate_portfolio=candidate_portfolio,
            )

            st.success(f"Materials generated!")
            st.write(f"Resume:       `{res_pdf}`")
            st.write(f"Cover letter: `{cl_pdf}`")
            st.rerun()
        except Exception as e:
            st.error(f"Generation failed: {e}")
            import traceback
            st.code(traceback.format_exc())
