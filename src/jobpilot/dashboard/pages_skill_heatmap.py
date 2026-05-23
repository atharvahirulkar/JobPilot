"""Skill Heatmap page — candidate model confidence scores visualization."""

import os
import streamlit as st
import pandas as pd
from jobpilot.candidate_model.model import CandidateModelManager


def render():
    st.header("Skill Heatmap")
    st.markdown("Your skill confidence scores over time (0.0–1.0).")

    db_url = st.session_state.get("db_url") or os.getenv(
        "DATABASE_URL", "sqlite:///jobpilot_w5.db"
    )

    try:
        mgr = CandidateModelManager(db_url=db_url)
        skills = mgr.get_all_skills()

        if not skills:
            st.info("No skills tracked yet. Run mock interview sessions to build your profile.")
            return

        # Convert to DataFrame for easier handling
        df = pd.DataFrame(skills)
        df = df.sort_values("confidence_score", ascending=False)

        # Display as bar chart
        st.bar_chart(
            df.set_index("skill_name")["confidence_score"],
            height=400,
            use_container_width=True,
        )

        st.markdown("---")
        st.subheader("Skill List")

        # Table view
        for _, row in df.iterrows():
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(row["skill_name"])
            with col2:
                confidence = row["confidence_score"]
                bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
                st.write(f"{bar} {confidence:.2f}")
            with col3:
                if confidence < 0.6:
                    st.warning("Weak")
                elif confidence < 0.8:
                    st.info("Dev")
                else:
                    st.success("Strong")

    except Exception as e:
        st.error(f"Error loading skills: {e}")
