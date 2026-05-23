"""Settings page — configure database, API keys, and candidate profile."""

import os
import streamlit as st


def render():
    st.header("Settings")
    st.markdown("Configure JobPilot for your environment.")

    st.subheader("Database")
    db_url = st.text_input(
        "Database URL",
        value=st.session_state.get("db_url") or os.getenv("DATABASE_URL", "sqlite:///jobpilot_w5.db"),
        help="PostgreSQL or SQLite connection string",
    )
    st.session_state["db_url"] = db_url

    st.subheader("Outputs")
    outputs_root = st.text_input(
        "Outputs root directory",
        value=st.session_state.get("outputs_root", "outputs/jobs"),
        help="Where tailored resumes and cover letters are saved",
    )
    st.session_state["outputs_root"] = outputs_root

    st.subheader("API Keys (optional)")
    st.markdown("""
    Leave blank to use environment variables or disable features.

    **LangSmith** (optional): Get key at https://smith.langchain.com/
    """)

    openai_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="For LLM-powered resume tailoring and cover letters",
    )

    langsmith_key = st.text_input(
        "LangSmith API Key",
        type="password",
        help="For tracing LangGraph nodes (optional)",
    )

    langsmith_project = st.text_input(
        "LangSmith Project Name",
        value=st.session_state.get("langsmith_project", "jobpilot"),
    )
    st.session_state["langsmith_project"] = langsmith_project

    st.subheader("Candidate Profile")
    candidate_name = st.text_input(
        "Your name",
        value=st.session_state.get("candidate_name", "Atharva Hirulkar"),
        help="Used in generated cover letters",
    )
    st.session_state["candidate_name"] = candidate_name

    candidate_email = st.text_input(
        "Your email",
        value=st.session_state.get("candidate_email", "[your.email@example.com]"),
    )
    st.session_state["candidate_email"] = candidate_email

    candidate_linkedin = st.text_input(
        "Your LinkedIn",
        value=st.session_state.get("candidate_linkedin", "linkedin.com/in/yourprofile"),
    )
    st.session_state["candidate_linkedin"] = candidate_linkedin

    st.markdown("---")

    if st.button("Save Settings"):
        st.success("Settings saved to session memory.")

    st.markdown("---")
    st.subheader("Next Steps")
    st.markdown(
        """
        1. **Populate jobs**: `python -m jobpilot.main run_pipeline --db-url ...`
        2. **Run interview**: `python -m jobpilot.main interview tmp_jobs.json`
        3. **View report**: Refresh this dashboard
        """
    )
