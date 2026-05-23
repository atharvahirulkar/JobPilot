"""Interview Session page — run or view mock interview sessions."""

import os
import streamlit as st
from sqlalchemy import create_engine, text
from jobpilot.db_models import init_db, InterviewAnswer


def render():
    st.header("Mock Interview")
    st.markdown("Run or review your mock interview sessions.")

    db_url = st.session_state.get("db_url") or os.getenv(
        "DATABASE_URL", "sqlite:///jobpilot_w5.db"
    )

    tab1, tab2 = st.tabs(["Start New Session", "Session History"])

    with tab1:
        st.info(
            "To start an interview session, run:\n\n"
            "`python -m jobpilot.main interview tmp_jobs.json --questions 4`\n\n"
            "Sessions will appear in the history below."
        )

    with tab2:
        try:
            SessionLocal = init_db(db_url)
            with SessionLocal() as session:
                answers = session.query(InterviewAnswer).order_by(InterviewAnswer.answered_at.desc()).limit(20).all()

            if not answers:
                st.info("No interview sessions yet.")
                return

            st.subheader("Recent Questions & Answers")
            for answer in answers:
                with st.expander(f"Q: {answer.question[:60]}..."):
                    st.write(f"**Question:** {answer.question}")
                    st.write(f"**Your Answer:** {answer.answer}")
                    if answer.score:
                        st.metric("Score", f"{answer.score}/10")
                    if answer.rationale:
                        import json
                        try:
                            feedback = json.loads(answer.rationale)
                            if feedback.get("strengths"):
                                st.success(f"Strengths: {', '.join(feedback['strengths'])}")
                            if feedback.get("weaknesses"):
                                st.warning(f"Gaps: {', '.join(feedback['weaknesses'])}")
                            if feedback.get("improved_answer"):
                                st.info(f"Next level: {feedback['improved_answer']}")
                        except Exception:
                            pass

        except Exception as e:
            st.error(f"Error loading sessions: {e}")
