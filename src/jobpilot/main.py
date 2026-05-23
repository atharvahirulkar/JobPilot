"""JobPilot CLI — W1–W6"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from .etl import ResumeETL
from .job_imports import load_jobs_file
from .jd_parser import JDParser
from .llm_parser import LLMJDParser
from .job_store import JobRepository
from .qdrant_setup import init_qdrant
from .scoring import ScoringEngine
from .scoring_agent import ScoringAgent
from .ranking import rank_jobs, print_ranked_summary
from .resume_tailor import LLMResumeTailor
from .cover_letter import LLMCoverLetterGenerator, build_request
from .pdf_generator import PDFGenerator
from .pipeline.graph import run_interview_session
from .agents.report_agent import ReportGenerator, EmailSender
from .scheduler.cron import run_pipeline, start_scheduler


def main():
    parser = argparse.ArgumentParser(prog="jobpilot", description="JobPilot CLI")
    sub = parser.add_subparsers(dest="cmd")

    p_etl = sub.add_parser("etl", help="Run resume ETL on a PDF")
    p_etl.add_argument("pdf", help="Path to resume PDF")

    p_jd = sub.add_parser("parse_jd", help="Parse a job description text file")
    p_jd.add_argument("jdfile", help="Path to JD text file")

    p_llm = sub.add_parser("parse_jd_llm", help="Parse a JD with a local-first LLM parser")
    p_llm.add_argument("jdfile", help="Path to JD text file")

    p_score = sub.add_parser("score", help="Score a resume against a JD locally")
    p_score.add_argument("resume", help="Path to resume text file")
    p_score.add_argument("jdfile", help="Path to JD text file")

    p_demo = sub.add_parser("demo", help="Run a small end-to-end ingest + rank demo")
    p_demo.add_argument("resume", help="Path to resume text file")
    p_demo.add_argument("jobs_file", help="Path to jobs JSON or CSV file")

    p_q = sub.add_parser("qdrant_init", help="Initialize qdrant collection")
    p_q.add_argument("--collection", default="jobpilot_resumes")

    # W3 commands
    p_score_llm = sub.add_parser("score_llm", help="[W3] Score resume against a JD using LLM alignment agent")
    p_score_llm.add_argument("resume", help="Path to resume text file")
    p_score_llm.add_argument("jdfile", help="Path to JD text file or JSON job dict")

    p_rank = sub.add_parser("rank", help="[W3] Score + rank all jobs in a JSON file against resume")
    p_rank.add_argument("resume", help="Path to resume text file")
    p_rank.add_argument("jobs_file", help="Path to jobs JSON file")
    p_rank.add_argument("--threshold", type=int, default=75, help="Top-match score threshold (default 75)")
    p_rank.add_argument("--limit", type=int, default=10, help="Max jobs to display")

    # W4 commands
    p_tailor = sub.add_parser("tailor_latex", help="[W4] Rewrite resume .tex bullets to match a JD")
    p_tailor.add_argument("template", help="Path to LaTeX resume template (.tex)")
    p_tailor.add_argument("jdfile", help="Path to JD JSON file or raw text")
    p_tailor.add_argument("--out", help="Output .tex path (default: stdout)")

    p_pdf = sub.add_parser("pdf", help="[W4] Tailor resume + cover letter → PDFs")
    p_pdf.add_argument("template", help="Path to LaTeX resume template (.tex)")
    p_pdf.add_argument("jdfile", help="Path to JD JSON file or raw text")
    p_pdf.add_argument("--resume-text", help="Path to plain-text resume (for cover letter bullets)")
    p_pdf.add_argument("--job-id", help="Job ID for output folder name (default: derived from company+title)")
    p_pdf.add_argument("--outputs-root", default="outputs/jobs", help="Root folder for outputs")

    # W5 commands
    p_interview = sub.add_parser("interview", help="[W5] Run a mock interview session for a job")
    p_interview.add_argument("jdfile", help="Path to JD JSON file or raw text")
    p_interview.add_argument("--questions", type=int, default=4, help="Number of questions (default 4)")
    p_interview.add_argument("--thread-id", default="default", help="Session thread ID (for resuming)")
    p_interview.add_argument("--db-url", help="PostgreSQL or SQLite URL (overrides DATABASE_URL env)")

    p_skills = sub.add_parser("skills", help="[W5] Show candidate skill confidence scores")
    p_skills.add_argument("--db-url", help="Database URL (overrides DATABASE_URL env)")

    # W6 commands
    p_report = sub.add_parser("report", help="[W6] Generate and print the morning report")
    p_report.add_argument("--db-url", help="Database URL")
    p_report.add_argument("--top-n", type=int, default=5)
    p_report.add_argument("--email", action="store_true", help="Send report via email")
    p_report.add_argument("--html", action="store_true", help="Print HTML instead of plain text")
    p_report.add_argument("--outputs-root", default="outputs/jobs")

    p_schedule = sub.add_parser("schedule", help="[W6] Start APScheduler 5AM cron")
    p_schedule.add_argument("--hour",    type=int, default=5)
    p_schedule.add_argument("--minute",  type=int, default=0)
    p_schedule.add_argument("--run-now", action="store_true", help="Run pipeline immediately then schedule")
    p_schedule.add_argument("--no-email", action="store_true")
    p_schedule.add_argument("--db-url")

    p_run = sub.add_parser("run_pipeline", help="[W6] Run the full morning pipeline once")
    p_run.add_argument("--db-url")
    p_run.add_argument("--email", action="store_true")
    p_run.add_argument("--top-n", type=int, default=5)

    # W7 command
    p_dashboard = sub.add_parser("dashboard", help="[W7] Launch Streamlit dashboard")
    p_dashboard.add_argument("--port", type=int, default=8501)
    p_dashboard.add_argument("--host", default="localhost")

    args = parser.parse_args()
    if args.cmd == "etl":
        etl = ResumeETL()
        text = etl.extract_text_from_pdf(args.pdf)
        parsed = etl.parse_resume(text)
        print(parsed)
    elif args.cmd == "parse_jd":
        text = Path(args.jdfile).read_text()
        parsed = JDParser().parse(text)
        print(parsed)
    elif args.cmd == "parse_jd_llm":
        text = Path(args.jdfile).read_text()
        parsed = LLMJDParser().parse(text)
        print(parsed)
    elif args.cmd == "score":
        resume_text = Path(args.resume).read_text()
        jd_text = Path(args.jdfile).read_text()
        jd = JDParser().parse(jd_text)
        result = ScoringEngine().score(resume_text, jd)
        print(result)
    elif args.cmd == "demo":
        resume_text = Path(args.resume).read_text()
        client = init_qdrant(collection_name="jobpilot_jobs")
        repo = JobRepository(client)
        jobs = load_jobs_file(args.jobs_file)
        repo.upsert_jobs(jobs)
        ranked = repo.rank_jobs_for_resume(resume_text, limit=5)
        for item in ranked:
            print(item)
    elif args.cmd == "qdrant_init":
        client = init_qdrant(collection_name=args.collection)
        print("Collections:", client.get_collections())

    # W3
    elif args.cmd == "score_llm":
        resume_text = Path(args.resume).read_text()
        raw = Path(args.jdfile).read_text()
        # Accept raw JD text, a JSON job dict, or a JSON list (takes first element)
        try:
            parsed = json.loads(raw)
            job = parsed[0] if isinstance(parsed, list) else parsed
        except json.JSONDecodeError:
            job = {"title": "", "company": "", "description": raw, "skills": []}
        agent = ScoringAgent()
        result = agent.score(resume_text, job)
        import pprint
        pprint.pprint(result.to_dict())
        print(f"\nis_top_match: {result.is_top_match}  (source: {result.source})")

    elif args.cmd == "rank":
        resume_text = Path(args.resume).read_text()
        jobs = load_jobs_file(args.jobs_file)
        agent = ScoringAgent()
        print(f"Scoring {len(jobs)} jobs against resume...")
        scores = agent.score_batch(resume_text, jobs)
        ranked = rank_jobs(jobs, scores)
        print_ranked_summary(ranked, limit=args.limit)

    # W4
    elif args.cmd == "tailor_latex":
        template_tex = Path(args.template).read_text(encoding="utf-8")
        raw = Path(args.jdfile).read_text()
        try:
            parsed = json.loads(raw)
            job = parsed[0] if isinstance(parsed, list) else parsed
        except json.JSONDecodeError:
            job = {"title": "", "company": "", "description": raw, "skills": []}
        tailor = LLMResumeTailor()
        result_tex = tailor.tailor_latex(template_tex, job)
        if args.out:
            Path(args.out).write_text(result_tex, encoding="utf-8")
            print(f"Tailored .tex written to {args.out}")
        else:
            print(result_tex)

    elif args.cmd == "pdf":
        template_tex = Path(args.template).read_text(encoding="utf-8")
        raw = Path(args.jdfile).read_text()
        try:
            parsed = json.loads(raw)
            job = parsed[0] if isinstance(parsed, list) else parsed
        except json.JSONDecodeError:
            job = {"title": "", "company": "", "description": raw, "skills": []}

        # Derive job_id from company + title if not supplied
        job_id = args.job_id or (
            f"{job.get('company','unknown')}_{job.get('title','role')}"
            .lower().replace(" ", "_")[:40]
        )

        # Tailor resume LaTeX
        tailor = LLMResumeTailor()
        tailored_tex = tailor.tailor_latex(template_tex, job)

        # Generate cover letter text
        resume_text = Path(args.resume_text).read_text() if args.resume_text else ""
        from .resume_tailor import ResumeTailor
        bullets = ResumeTailor().tailor(resume_text, job).bullets if resume_text else []
        cl_request = build_request("Atharva Hirulkar", job, bullets)
        cl_gen = LLMCoverLetterGenerator()
        cl_text = cl_gen.generate(cl_request)

        # Compile to PDFs
        gen = PDFGenerator(outputs_root=args.outputs_root)
        resume_pdf  = gen.resume_pdf(tailored_tex, job_id)
        cl_pdf      = gen.cover_letter_pdf(cl_text, job, job_id)

        print(f"Resume PDF    : {resume_pdf}")
        print(f"Cover letter  : {cl_pdf}")

    # W6
    elif args.cmd == "report":
        gen    = ReportGenerator(db_url=args.db_url, outputs_root=args.outputs_root)
        report = gen.generate(top_n=args.top_n)
        print(report.to_html() if args.html else report.to_text())
        if args.email:
            EmailSender().send(report)
            print("\nEmail sent.")

    elif args.cmd == "run_pipeline":
        import pprint
        summary = run_pipeline(db_url=args.db_url, send_email=args.email, top_n=args.top_n)
        pprint.pprint(summary)

    elif args.cmd == "schedule":
        start_scheduler(
            hour=args.hour,
            minute=args.minute,
            db_url=args.db_url,
            send_email=not args.no_email,
            run_now_also=args.run_now,
        )

    # W7
    elif args.cmd == "dashboard":
        import subprocess
        app_path = Path(__file__).parent / "dashboard" / "app.py"
        subprocess.run([
            "streamlit", "run", str(app_path),
            "--server.port", str(args.port),
            "--server.address", args.host,
        ])

    # W5
    elif args.cmd == "interview":
        raw = Path(args.jdfile).read_text()
        try:
            parsed = json.loads(raw)
            job = parsed[0] if isinstance(parsed, list) else parsed
        except json.JSONDecodeError:
            job = {"title": "", "company": "", "description": raw, "skills": []}
        run_interview_session(
            job=job,
            n_questions=args.questions,
            db_url=args.db_url,
            thread_id=args.thread_id,
        )

    elif args.cmd == "skills":
        from .candidate_model.model import CandidateModelManager
        mgr = CandidateModelManager(db_url=args.db_url)
        skills = mgr.get_all_skills()
        if not skills:
            print("No skills tracked yet. Run an interview session first.")
        else:
            print(f"\n{'Skill':<30} {'Confidence':>10}  Bar")
            print("─" * 55)
            for s in skills:
                conf = s["confidence_score"]
                bar  = "█" * int(conf * 10) + "░" * (10 - int(conf * 10))
                flag = " ← weak" if conf < 0.6 else ""
                print(f"{s['skill_name']:<30} {conf:>10.2f}  {bar}{flag}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
