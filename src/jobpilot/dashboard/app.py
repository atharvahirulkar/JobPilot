"""JobPilot Streamlit Dashboard — multi-page UI for job tracking, skill monitoring, and interview prep.

Pages:
  - Job Tracker: ranked jobs with scores, gaps, materials status
  - Skill Heatmap: candidate model confidence scores over time
  - Interview Session: run/view mock interview with real-time scores
  - Morning Report: generated daily report
  - Settings: configure database, API keys, candidate profile
"""

import os
from pathlib import Path

import streamlit as st

# Configure page
st.set_page_config(
    page_title="JobPilot",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar navigation
st.sidebar.title("JobPilot Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    options=[
        "Job Tracker",
        "Skill Heatmap",
        "Interview Session",
        "Morning Report",
        "Settings",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **JobPilot** — Autonomous career agent.

    Searches jobs · Scores matches · Tailors resume · Runs interviews.
    """
)

# Load pages
if page == "Job Tracker":
    from jobpilot.dashboard import pages_job_tracker
    pages_job_tracker.render()

elif page == "Skill Heatmap":
    from jobpilot.dashboard import pages_skill_heatmap
    pages_skill_heatmap.render()

elif page == "Interview Session":
    from jobpilot.dashboard import pages_interview
    pages_interview.render()

elif page == "Morning Report":
    from jobpilot.dashboard import pages_report
    pages_report.render()

elif page == "Settings":
    from jobpilot.dashboard import pages_settings
    pages_settings.render()
