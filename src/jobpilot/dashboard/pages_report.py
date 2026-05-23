"""Morning Report page — display the generated morning report."""

import os
import streamlit as st
from jobpilot.agents.report_agent import ReportGenerator


def render():
    st.header("Morning Report")
    st.markdown("Today's job brief — new matches, skill gaps, and prep status.")

    db_url = st.session_state.get("db_url") or os.getenv(
        "DATABASE_URL", "sqlite:///jobpilot_w5.db"
    )
    outputs_root = st.session_state.get("outputs_root", "outputs/jobs")

    col1, col2 = st.columns(2)
    with col1:
        top_n = st.slider("Top N jobs", min_value=1, max_value=20, value=5)
    with col2:
        min_score = st.slider("Minimum score", min_value=0, max_value=100, value=0, step=5)

    try:
        gen = ReportGenerator(db_url=db_url, outputs_root=outputs_root)
        report = gen.generate(top_n=top_n, min_score=min_score)

        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Scored", report.total_scored)
        with col2:
            st.metric("Strong Matches", report.top_matches_count)
        with col3:
            st.metric("Report Date", report.report_date)

        st.markdown("---")

        # Display jobs
        st.subheader("Top Matches")
        for job in report.jobs:
            with st.expander(
                f"#{job.rank} {job.title} @ {job.company} ({job.match_score}%)"
            ):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Match Score", f"{job.match_score}/100")
                    st.metric("Role Fit", job.role_fit)
                with col2:
                    st.metric("Company Tier", job.company_tier)
                    st.metric("Location", job.location)

                if job.skill_gaps:
                    st.write(f"**Skill Gaps:** {', '.join(job.skill_gaps)}")

                materials = []
                if job.resume_ready:
                    materials.append("Resume")
                if job.cover_letter_ready:
                    materials.append("Cover Letter")
                if materials:
                    st.success(f"Materials ready: {', '.join(materials)}")
                else:
                    st.info("Materials not yet generated")

        st.markdown("---")

        # Recurring gaps
        if report.recurring_gaps:
            st.subheader("Recurring Skill Gaps")
            st.write(", ".join(report.recurring_gaps[:5]))

        # Weak skills
        if report.weak_skills:
            st.subheader("Interview Focus Areas")
            for skill in report.weak_skills:
                st.write(f"• {skill}")

        st.markdown("---")

        # Plain text version
        with st.expander("View as plain text"):
            st.code(report.to_text())

    except Exception as e:
        st.error(f"Error generating report: {e}")
