"""Settings page — configure database, API keys, and candidate profile."""

import os

import streamlit as st


def render():
    st.header("Settings")
    st.markdown("Configure JobPilot for your environment. Changes are saved to the current session.")

    # ── Database ────────────────────────────────────────────────────────────
    st.subheader("Database")
    db_url = st.text_input(
        "Database URL",
        value=st.session_state.get("db_url") or os.getenv(
            "DATABASE_URL", "sqlite:///jobpilot_w5.db"
        ),
        help="PostgreSQL: postgresql://user:pass@host/db  |  SQLite: sqlite:///path/to/file.db",
    )
    st.session_state["db_url"] = db_url

    # ── Outputs ─────────────────────────────────────────────────────────────
    st.subheader("Outputs")
    outputs_root = st.text_input(
        "Outputs root directory",
        value=st.session_state.get("outputs_root", "outputs/jobs"),
        help="Where tailored resumes and cover letters are saved",
    )
    st.session_state["outputs_root"] = outputs_root

    # ── API Keys ────────────────────────────────────────────────────────────
    st.subheader("API Keys")
    st.markdown(
        "Leave blank to use values from your `.env` file / environment variables."
    )

    openai_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.get("openai_key", ""),
        help="Used for LLM scoring, resume tailoring, cover letters, and interview evaluation.",
    )
    if openai_key:
        st.session_state["openai_key"] = openai_key
        os.environ["OPENAI_API_KEY"]   = openai_key

    serpapi_key = st.text_input(
        "SerpAPI Key",
        type="password",
        value=st.session_state.get("serpapi_key", ""),
        help="Used for job search via Google Jobs / SerpAPI.",
    )
    if serpapi_key:
        st.session_state["serpapi_key"]  = serpapi_key
        os.environ["SERPAPI_API_KEY"]    = serpapi_key

    langsmith_key = st.text_input(
        "LangSmith API Key",
        type="password",
        value=st.session_state.get("langsmith_key", ""),
        help="Optional — enables LangSmith tracing of LangGraph nodes.",
    )
    if langsmith_key:
        st.session_state["langsmith_key"]   = langsmith_key
        os.environ["LANGSMITH_API_KEY"]     = langsmith_key

    langsmith_project = st.text_input(
        "LangSmith Project Name",
        value=st.session_state.get("langsmith_project", os.getenv("LANGSMITH_PROJECT", "jobpilot")),
    )
    st.session_state["langsmith_project"] = langsmith_project

    # ── Candidate Profile ───────────────────────────────────────────────────
    st.subheader("Candidate Profile")
    st.markdown("Used in generated cover letters and PDFs.")

    candidate_name = st.text_input(
        "Your name",
        value=st.session_state.get("candidate_name") or os.getenv("CANDIDATE_NAME", ""),
        help="Appears in the cover letter header and PDF file names.",
    )
    st.session_state["candidate_name"] = candidate_name

    candidate_email = st.text_input(
        "Your email",
        value=st.session_state.get("candidate_email") or os.getenv("CANDIDATE_EMAIL", ""),
    )
    st.session_state["candidate_email"] = candidate_email

    candidate_phone = st.text_input(
        "Your phone number",
        value=st.session_state.get("candidate_phone") or os.getenv("CANDIDATE_PHONE", ""),
        help="Optional. Included in the cover letter header.",
    )
    st.session_state["candidate_phone"] = candidate_phone

    candidate_linkedin = st.text_input(
        "Your LinkedIn URL",
        value=st.session_state.get("candidate_linkedin") or os.getenv("CANDIDATE_LINKEDIN", ""),
        placeholder="linkedin.com/in/yourname",
    )
    st.session_state["candidate_linkedin"] = candidate_linkedin

    candidate_portfolio = st.text_input(
        "Your portfolio / GitHub URL",
        value=st.session_state.get("candidate_portfolio") or os.getenv("CANDIDATE_PORTFOLIO", ""),
        placeholder="github.com/yourname  or  yourname.dev",
        help="Optional. Renders as a clickable link in the cover letter PDF.",
    )
    st.session_state["candidate_portfolio"] = candidate_portfolio

    resume_path = st.text_input(
        "Resume file path",
        value=st.session_state.get("resume_path") or os.getenv("RESUME_PATH", "data/master_resume.txt"),
        help="Path to your master resume (.txt or .pdf). Used for LLM scoring and tailoring.",
    )
    st.session_state["resume_path"] = resume_path
    if resume_path:
        os.environ["RESUME_PATH"] = resume_path

    st.markdown("---")

    if st.button("Save Settings", type="primary"):
        st.success("Settings saved to session. They will persist for this browser tab.")

    # ── Environment check ───────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Environment Check")

    checks = {
        "DATABASE_URL":        os.getenv("DATABASE_URL"),
        "OPENAI_API_KEY":      os.getenv("OPENAI_API_KEY"),
        "SERPAPI_API_KEY":     os.getenv("SERPAPI_API_KEY"),
        "RESUME_PATH":         os.getenv("RESUME_PATH"),
        "CANDIDATE_NAME":      os.getenv("CANDIDATE_NAME"),
        "CANDIDATE_EMAIL":     os.getenv("CANDIDATE_EMAIL"),
        "CANDIDATE_LINKEDIN":  os.getenv("CANDIDATE_LINKEDIN"),
        "CANDIDATE_PORTFOLIO": os.getenv("CANDIDATE_PORTFOLIO"),
    }

    for key, val in checks.items():
        if val:
            st.success(f"✓  `{key}` is set")
        else:
            st.warning(f"⚠  `{key}` is not set")

    st.markdown("---")
    st.subheader("Next Steps")
    st.markdown(
        """
        1. Set your **Database URL** above and confirm the environment check shows green.
        2. Go to **Controls → Import Jobs** to load a jobs file into the database.
        3. Go to **Controls → Run Pipeline** to score jobs, tailor resumes, and generate the report.
        4. Go to **Job Tracker** to browse results and trigger per-job material generation.
        5. Go to **Interview Session** to run a mock interview against your top job.
        6. Go to **Skill Heatmap** to see how your confidence scores evolve over time.
        """
    )
