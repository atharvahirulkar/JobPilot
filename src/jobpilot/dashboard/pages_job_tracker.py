"""Job Tracker page — ranked jobs with scores, skill gaps, materials status."""

import os
import streamlit as st
from jobpilot.db_repository import JobRepository


def render():
    st.header("Job Tracker")
    st.markdown("Ranked jobs with match scores, skill gaps, and material generation status.")

    # Get database URL
    db_url = st.session_state.get("db_url") or os.getenv(
        "DATABASE_URL", "sqlite:///jobpilot_w5.db"
    )

    try:
        repo = JobRepository(db_url=db_url)
        jobs = repo.get_top_jobs_with_scores(limit=50, min_score=0)

        if not jobs:
            st.info("No jobs in database yet. Run `python -m jobpilot.main run_pipeline` to populate.")
            return

        # Display as table
        st.dataframe(
            [
                {
                    "Rank": i + 1,
                    "Title": j["title"],
                    "Company": j["company"],
                    "Score": f"{j['match_score']}%",
                    "Fit": j["role_fit"],
                    "Tier": j["company_tier"],
                    "Gaps": ", ".join(eval(j["skill_gaps"])[:2]) if j["skill_gaps"] else "-",
                }
                for i, j in enumerate(jobs)
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("---")
        st.subheader("Details")

        # Show details for selected job
        selected_idx = st.number_input(
            "View details for job (rank number)", min_value=1, max_value=len(jobs), value=1
        ) - 1
        job = jobs[selected_idx]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Match Score", f"{job['match_score']}/100")
            st.metric("Role Fit", job["role_fit"])

        with col2:
            st.metric("Company Tier", job["company_tier"])
            st.metric("Source", job["source"])

        st.markdown(f"**URL:** {job['url']}")

        gaps = eval(job["skill_gaps"]) if job["skill_gaps"] else []
        if gaps:
            st.markdown(f"**Skill Gaps:** {', '.join(gaps)}")

    except Exception as e:
        st.error(f"Error loading jobs: {e}")
