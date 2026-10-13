"""Interview Session page — inline mock interview powered by LangGraph.

State machine (stored in st.session_state):
  idle    → user picks a job + question count → Start Interview
  running → graph interrupted at current question → user types answer → Submit
  done    → graph finished → show summary + per-question breakdown

The LangGraph graph uses interrupt() for human-in-the-loop.  Each Streamlit
rerun resumes exactly where the graph left off via the MemorySaver checkpointer.

Voice mode (optional, needs ELEVENLABS_API_KEY): the interviewer reads each
question aloud via ElevenLabs TTS, the candidate can answer by voice
(transcribed with ElevenLabs Scribe into an editable transcript), and feedback
plus a session wrap-up are spoken back. Any voice failure falls back to text.
"""

import hashlib
import json
import os
import traceback

import streamlit as st


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _db_url() -> str:
    return st.session_state.get("db_url") or os.getenv(
        "DATABASE_URL", "sqlite:///jobpilot_w5.db"
    )


def _load_jobs() -> list:
    try:
        from jobpilot.db_repository import JobRepository
        return JobRepository(db_url=_db_url()).get_top_jobs_with_scores(limit=60, min_score=0)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# LangGraph session management
# ---------------------------------------------------------------------------

def _voice_available() -> bool:
    return bool(os.getenv("ELEVENLABS_API_KEY"))


def _start_session(job: dict, n_questions: int, voice: bool = False) -> None:
    """Build the graph, kick off the first node, park at first question."""
    import uuid
    from jobpilot.pipeline.graph import build_graph

    graph  = build_graph()
    config = {"configurable": {"thread_id": f"st_{uuid.uuid4().hex[:10]}"}}

    initial_state = {
        "current_job":          job,
        "generated_questions":  [],
        "current_q_index":      0,
        "user_answers":         [],
        "evaluation_results":   [],
        "candidate_profile":    {},
        "session_summary":      {},
        "n_questions":          n_questions,
        "db_url":               _db_url(),
        "api_key":              os.getenv("OPENAI_API_KEY"),
    }

    # Run until first interrupt
    graph.invoke(initial_state, config)

    snap          = graph.get_state(config)
    interrupt_val = _extract_interrupt(snap)

    voice_agent, prefetch_stats = None, {}
    if voice:
        from jobpilot.voice import VoiceInterviewer

        voice_agent = VoiceInterviewer(db_url=_db_url(), session_id=config["configurable"]["thread_id"])
        if voice_agent.enabled:
            # Synthesize every question up front, in parallel, so playback never waits.
            prefetch_stats = voice_agent.prefetch(snap.values.get("generated_questions", []), job)
        else:
            voice_agent = None

    st.session_state.update({
        "iv_voice":    voice_agent,
        "iv_prefetch": prefetch_stats,
        "iv_tx":       {},
        "iv_state":    "running",
        "iv_graph":    graph,
        "iv_config":   config,
        "iv_job":      job,
        "iv_question": interrupt_val.get("question", ""),
        "iv_q_index":  interrupt_val.get("index", 0),
        "iv_q_total":  interrupt_val.get("total", n_questions),
        "iv_results":  [],
        "iv_summary":  {},
    })


def _submit_answer(answer: str) -> None:
    """Resume the graph with the user's answer; advance state."""
    from langgraph.types import Command

    graph  = st.session_state["iv_graph"]
    config = st.session_state["iv_config"]

    graph.invoke(Command(resume=answer), config)

    snap    = graph.get_state(config)
    results = snap.values.get("evaluation_results", [])
    st.session_state["iv_results"] = results

    if not snap.next:
        # Session finished
        st.session_state["iv_summary"] = snap.values.get("session_summary", {})
        st.session_state["iv_state"]   = "done"
    else:
        # More questions
        interrupt_val = _extract_interrupt(snap)
        st.session_state["iv_question"] = interrupt_val.get("question", "")
        st.session_state["iv_q_index"]  = interrupt_val.get("index", 0)


def _extract_interrupt(snap) -> dict:
    """Pull the interrupt payload out of a graph snapshot."""
    try:
        task = snap.tasks[0] if snap.tasks else None
        return task.interrupts[0].value if (task and task.interrupts) else {}
    except Exception:
        return {}


def _reset_session() -> None:
    for key in ("iv_state", "iv_graph", "iv_config", "iv_job",
                "iv_question", "iv_q_index", "iv_q_total",
                "iv_results", "iv_summary", "iv_voice", "iv_prefetch", "iv_tx",
                "iv_played"):
        st.session_state.pop(key, None)


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render():
    st.header("Mock Interview")

    tab_live, tab_history = st.tabs(["Live Session", "Session History"])

    with tab_live:
        _render_live()

    with tab_history:
        _render_history()


# ---------------------------------------------------------------------------
# Live session tab
# ---------------------------------------------------------------------------

def _render_live():
    state = st.session_state.get("iv_state", "idle")

    if state == "idle":
        _render_idle()
    elif state == "running":
        _render_question()
    elif state == "done":
        _render_summary()


def _render_idle():
    st.markdown(
        "Pick a job from the database, choose the number of questions, "
        "and click **Start Interview**. The session runs entirely in the dashboard."
    )

    jobs = _load_jobs()
    if not jobs:
        st.warning(
            "No jobs in the database yet. "
            "Go to **Controls → Import Jobs** or **Run Pipeline** to populate it first."
        )
        return

    labels = [
        f"{j['title']} @ {j['company']}  ({j['match_score']}%)"
        for j in jobs
    ]
    chosen_label = st.selectbox("Select job for your interview", labels)
    chosen_job   = jobs[labels.index(chosen_label)]

    col1, col2 = st.columns(2)
    with col1:
        n_q = st.slider("Number of questions", min_value=2, max_value=8, value=4)
        voice = st.toggle(
            "Voice mode (ElevenLabs)",
            value=_voice_available(),
            disabled=not _voice_available(),
            help="Interviewer speaks each question; answer out loud or type. "
                 "Set ELEVENLABS_API_KEY in .env to enable.",
        )
    with col2:
        st.markdown("")
        st.markdown("")
        if st.button("Start Interview", type="primary", use_container_width=True):
            label = "Generating questions and preparing the interviewer's voice…" if voice \
                else "Generating tailored questions…"
            with st.spinner(label):
                try:
                    _start_session(chosen_job, n_q, voice=voice)
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not start session: {e}")
                    st.code(traceback.format_exc())


def _render_question():
    job      = st.session_state.get("iv_job", {})
    q_idx    = st.session_state.get("iv_q_index", 0)
    q_total  = st.session_state.get("iv_q_total", 4)
    question = st.session_state.get("iv_question", "")
    results  = st.session_state.get("iv_results", [])

    # Header bar
    st.markdown(f"#### {job.get('title', '?')} @ {job.get('company', '?')}")
    st.progress(q_idx / q_total, text=f"Question {q_idx + 1} of {q_total}")

    # Previous answers (collapsed)
    if results:
        with st.expander(f"Previous answers — {len(results)} scored", expanded=False):
            for i, r in enumerate(results):
                score = r.get("overall_score", 0)
                cols  = st.columns([1, 4])
                cols[0].metric(f"Q{i + 1}", f"{score:.1f}/10")
                with cols[1]:
                    if r.get("strengths"):
                        st.success(f"✓  {r['strengths'][0]}")
                    if r.get("weaknesses"):
                        st.warning(f"✗  {r['weaknesses'][0]}")

    voice = st.session_state.get("iv_voice")

    # Spoken feedback on the previous answer (click to play, so it never
    # talks over the next question).
    if voice and results:
        fb_audio = voice.speak_feedback(results[-1])
        if fb_audio:
            st.caption(f"🔊 Feedback on Q{len(results)}")
            st.audio(fb_audio, format="audio/mpeg")

    st.markdown("---")

    # Current question
    st.subheader(f"Q{q_idx + 1}:  {question}")

    if voice:
        q_audio = voice.speak_question(question, q_idx, q_total, job)  # cache hit after prefetch
        if q_audio:
            # Autoplay once per question; later reruns (recording, typing) stay quiet.
            played = st.session_state.setdefault("iv_played", set())
            st.audio(q_audio, format="audio/mpeg", autoplay=q_idx not in played)
            played.add(q_idx)
        elif voice.last_error:
            st.caption(f"Voice unavailable, continuing in text: {voice.last_error}")

        if hasattr(st, "audio_input"):
            recording = st.audio_input("Answer out loud (or type below)", key=f"iv_rec_{q_idx}")
            if recording is not None:
                audio_bytes = recording.getvalue()
                digest = hashlib.sha256(audio_bytes).hexdigest()
                tx_cache = st.session_state.setdefault("iv_tx", {})
                if tx_cache.get(q_idx, (None,))[0] != digest:
                    with st.spinner("Transcribing…"):
                        text = voice.transcribe(audio_bytes, filename=recording.name or "answer.wav")
                    tx_cache[q_idx] = (digest, text)
                    if text:
                        st.session_state[f"iv_answer_{q_idx}"] = text
                    else:
                        st.warning("Couldn't transcribe that recording. Try again or type your answer.")

    answer = st.text_area(
        "Your answer",
        key=f"iv_answer_{q_idx}",
        height=180,
        placeholder="Type your answer here…",
    )

    col_submit, col_abandon, _ = st.columns([1, 1, 4])
    with col_submit:
        if st.button("Submit →", type="primary", disabled=not answer.strip()):
            with st.spinner("Evaluating…"):
                try:
                    _submit_answer(answer.strip())
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.code(traceback.format_exc())
    with col_abandon:
        if st.button("Abandon", type="secondary"):
            _reset_session()
            st.rerun()


def _render_summary():
    summary = st.session_state.get("iv_summary", {})
    results = st.session_state.get("iv_results", [])
    job     = st.session_state.get("iv_job", {})

    st.success("Session complete!")
    st.markdown(f"#### {job.get('title', '?')} @ {job.get('company', '?')}")

    avg   = summary.get("session_score", 0)
    updated = summary.get("updated", [])

    voice = st.session_state.get("iv_voice")
    if voice:
        wrap = f"That's the end of the session. You averaged {avg:.1f} out of 10 across {len(results)} questions."
        weakest = min(results, key=lambda r: r.get("overall_score", 0)) if results else None
        if weakest and weakest.get("weaknesses"):
            wrap += f" The main thing to work on: {weakest['weaknesses'][0]}"
        wrap_audio = voice.speak(wrap)
        if wrap_audio:
            st.audio(wrap_audio, format="audio/mpeg", autoplay=True)
        _render_voice_stats(voice.session_id)

    c1, c2, c3 = st.columns(3)
    c1.metric("Average Score",   f"{avg:.1f} / 10")
    c2.metric("Questions",       len(results))
    c3.metric("Skills Updated",  len(updated))

    if updated:
        st.info(f"Candidate model updated for:  {', '.join(updated)}")

    st.markdown("---")
    st.subheader("Question Breakdown")

    for i, r in enumerate(results):
        score = r.get("overall_score", 0)
        bar   = "█" * int(score) + "░" * (10 - int(score))
        with st.expander(f"Q{i + 1}  |  {score:.1f}/10  [{bar}]", expanded=(i == 0)):
            cols = st.columns(3)
            cols[0].metric("Relevance", r.get("relevance", "-"))
            cols[1].metric("Depth",     r.get("depth", "-"))
            cols[2].metric("Accuracy",  r.get("accuracy", "-"))

            if r.get("strengths"):
                st.success(f"✓  Strengths:  {',  '.join(r['strengths'])}")
            if r.get("weaknesses"):
                st.warning(f"✗  Improve:    {',  '.join(r['weaknesses'])}")
            if r.get("improved_answer"):
                st.info(f"💡  Next-level answer:  {r['improved_answer']}")

    st.markdown("---")
    if st.button("Start Another Session", type="primary"):
        _reset_session()
        st.rerun()


def _render_voice_stats(session_id: str) -> None:
    """Latency + cost telemetry for this session's ElevenLabs calls."""
    try:
        from jobpilot.voice.telemetry import load_metrics, summarize

        metrics = [m for m in load_metrics(_db_url()) if m.session_id == session_id]
        if not metrics:
            return
        s = summarize(metrics)
        with st.expander("Voice telemetry (this session)"):
            c1, c2, c3, c4 = st.columns(4)
            p50 = s["tts_ttfb_p50_ms"]
            p95 = s["tts_ttfb_p95_ms"]
            c1.metric("TTS first audio p50", f"{p50:.0f} ms" if p50 is not None else "-")
            c2.metric("TTS first audio p95", f"{p95:.0f} ms" if p95 is not None else "-")
            hit = s["tts_cache_hit_rate"]
            c3.metric("Cache hit rate", f"{hit:.0%}" if hit is not None else "-")
            c4.metric("Chars billed", s["tts_chars_billed"])
            prefetch = st.session_state.get("iv_prefetch") or {}
            st.caption(
                f"Prefetched {prefetch.get('ok', 0)} question clips in parallel "
                f"({prefetch.get('failed', 0)} failed) · STT calls: {s['stt_calls']} · errors: {s['errors']}"
            )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# History tab
# ---------------------------------------------------------------------------

def _render_history():
    try:
        from jobpilot.db_models import init_db, InterviewAnswer

        SessionLocal = init_db(_db_url())
        with SessionLocal() as session:
            answers = (
                session.query(InterviewAnswer)
                .order_by(InterviewAnswer.answered_at.desc())
                .limit(40)
                .all()
            )
            # Detach from session before closing
            answers = [
                {
                    "question":    a.question,
                    "answer":      a.answer,
                    "score":       a.score,
                    "rationale":   a.rationale,
                    "answered_at": a.answered_at,
                }
                for a in answers
            ]

        if not answers:
            st.info("No interview sessions recorded yet. Start one in the **Live Session** tab.")
            return

        st.subheader(f"Last {len(answers)} answered questions")

        for a in answers:
            ts    = a["answered_at"].strftime("%b %d %H:%M") if a["answered_at"] else ""
            label = f"{a['question'][:70]}…   |   Score: {a['score'] or '?'}/10   |   {ts}"
            with st.expander(label):
                st.markdown(f"**Question:** {a['question']}")
                st.markdown(f"**Answer:** {a['answer']}")
                if a["score"]:
                    st.metric("Score", f"{a['score']}/10")
                if a["rationale"]:
                    try:
                        fb = json.loads(a["rationale"])
                        if fb.get("strengths"):
                            st.success(f"Strengths: {', '.join(fb['strengths'])}")
                        if fb.get("weaknesses"):
                            st.warning(f"Areas to improve: {', '.join(fb['weaknesses'])}")
                        if fb.get("improved_answer"):
                            st.info(f"💡 Next level: {fb['improved_answer']}")
                    except Exception:
                        pass

    except Exception as e:
        st.error(f"Error loading history: {e}")
        st.code(traceback.format_exc())
