# 🤖 JobPilot

> **Your autonomous career agent.** Searches while you sleep. Preps while you wake.

![Status](https://img.shields.io/badge/status-operational-brightgreen)
![Python](https://img.shields.io/badge/python-3.11-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2-purple)
![Deployment](https://img.shields.io/badge/Deployment-Local-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## The Problem

Job searching is broken. You spend hours manually scanning portals, copy-pasting your resume, rewriting cover letters, and still walk into interviews unprepared for the specific role. It's slow, repetitive, and demoralizing.

**JobPilot automates all of it.**

---

## What It Does

Run on-demand, or schedule a 5 AM cron - JobPilot runs the full pipeline autonomously while you sleep:

```
🔍  Scrapes 412 curated target companies for new DS/MLE/SWE/Quant roles
      Direct ATS JSON APIs (Greenhouse · Lever · Ashby · Workable) + Playwright fallback for custom career pages

📊  Scores every JD against your resume using an LLM alignment agent
      Match score 0–100 · Skill gaps · Role fit · Company tier

✍️  For top matches (configurable threshold), tailors your resume + writes a cover letter
      Reorders bullets by relevance · Injects keywords naturally
      Outputs: tailored_resume.pdf + cover_letter.pdf per job

📋  Compiles your morning report
      Top 5 jobs · Materials ready · Gaps flagged · Prep status

🎯  Auto-queues a personalized mock interview session
      Questions weighted by YOUR weakest skill areas for today's top job

🎙️  Runs it as a spoken interview (optional, ElevenLabs)
      Interviewer reads questions aloud · you answer out loud · spoken feedback

📧  Delivers everything
      Email digest + Streamlit dashboard - ready when you wake up
```

By the time you have your first coffee, you have a ranked shortlist of roles,
tailored application materials, and a mock interview session loaded for your top pick.

---

## Demo

**Live Dashboard - Run Locally:**

```bash
python -m jobpilot.main dashboard
# → http://localhost:8502
```

**Morning Report Example (output varies per run as new jobs come in):**
```
Good morning. Here's today's job brief.

📊 Searched 412 target companies → scored N new roles.
🎯 K strong matches surfaced.

Top Pick: <role> @ <company>
  Match Score: 85% | Gap: <recurring skill gaps>
  📄 Tailored resume → outputs/jobs/<job_id>/tailored_resume.pdf
  📝 Cover letter   → outputs/jobs/<job_id>/cover_letter.pdf

🎤 Mock interview session queued.
   Focus: questions weighted to your top match's JD
   Open dashboard → http://localhost:8502
```

### Screenshots

> _Add real screenshots from your latest run here:_

```
docs/screenshots/
  ├── job_tracker.png       # Job Tracker page with ranked matches
  ├── morning_report.png    # Morning Report summary
  └── mock_interview.png    # Live mock-interview session
```

---

## Architecture

```
┌─────────────────────────────────┐
│  APScheduler (5 AM cron) / CLI  │
└────────────────┬────────────────┘
                 │
┌────────────────▼────────────────┐
│       LangGraph Pipeline        │
│   (Structured State Machine)    │
└──┬──────────┬────────┬──────────┘
   │          │        │
┌──▼──────┐ ┌─▼──────┐ ┌─▼──────────────┐
│  Job    │ │Scoring │ │ Resume Tailor  │
│ Search  │ │Engine  │ │ + Cover Letter │
│ Agent   │ │        │ │   Generator    │
│         │ │LLM     │ │                │
│412 cos. │ │0-100 + │ │LaTeX → PDF     │
│Playwright│ │gaps    │ │via pdflatex    │
│+ATS APIs│ │        │ │   (optional)   │
└────┬────┘ └───┬────┘ └───────┬────────┘
     │          │              │
┌────▼──────────▼──────────────▼────────┐
│   SQLite (default) or PostgreSQL      │
│  jobs · scores · candidate model      │
└───────────────────┬───────────────────┘
                    │
        ┌───────────▼───────────┐
        │         Qdrant        │
        │  Hybrid dense + BM25  │
        │  Resume + JD chunks   │
        └───────────┬───────────┘
                    │
        ┌───────────▼───────────────┐
        │   Mock Interview Engine   │
        │   (LangGraph interrupt)   │
        │                           │
        │ Question Gen → Answer     │
        │ → LLM-as-judge → Update   │
        │   Candidate Model         │
        └───────────┬───────────────┘
                    │
        ┌───────────▼───────────┐
        │  Morning Report +     │
        │  Streamlit Dashboard  │
        │  Email (smtplib)      │
        └───────────────────────┘
```

---

## What Makes This Different

| Feature | JobPilot | Standard job boards |
|---|---|---|
| Automated daily search | ✅ 5 AM cron | ❌ Manual |
| Resume tailored per job | ✅ LLM alignment agent | ❌ One-size-fits-all |
| Cover letter per job | ✅ Generated, not templated | ❌ Manual |
| Skill gap analysis | ✅ Per JD, every morning | ❌ None |
| Persistent candidate memory | ✅ Improves over sessions | ❌ None |
| Spoken mock interviews | ✅ ElevenLabs TTS + Scribe STT | ❌ None |
| Interview prep per role | ✅ Auto-queued | ❌ None |
| Full observability | ✅ LangSmith traces | ❌ N/A |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent framework | LangGraph - structured state machine with typed state |
| LLM | GPT-4o-mini (prod) · Claude Haiku (fallback) |
| Vector store | Qdrant - hybrid dense (all-MiniLM-L6-v2) + BM25 sparse |
| RAG evaluation | RAGAS - faithfulness · relevance · answer quality |
| Database | PostgreSQL - jobs · candidate model · answer history |
| ETL | PyPDF2 + spaCy + LLM extraction |
| Job scraping | Playwright + free ATS JSON APIs (Greenhouse · Lever · Ashby · Workable) - 412 curated companies |
| PDF generation | LaTeX → `pdflatex` (optional; pipeline skips PDFs gracefully if absent) |
| Observability | LangSmith - every agent node traced end-to-end |
| Voice | ElevenLabs Flash v2.5 (streaming TTS) · Scribe (speech-to-text) · content-addressed audio cache |
| Frontend | Streamlit - job tracker · skill heatmap · mock interview UI · morning report |
| Scheduler | APScheduler - 5 AM cron (local) + manual CLI |
| Deployment | Local (Linux/macOS) or Docker |
| Language | Python 3.11 |

---

## Project Structure

```
jobpilot/
├── src/jobpilot/
│   ├── agents/
│   │   ├── job_search_agent.py      # Multi-portal scraper + LLM extractor
│   │   ├── scoring_agent.py         # Resume↔JD alignment + gap extraction
│   │   ├── interview_agent.py       # Question gen + LLM-as-judge evaluator
│   │   └── report_agent.py          # Morning report + email sender
│   ├── rag/
│   │   ├── ingest.py                # Resume + JD → Qdrant (hybrid index)
│   │   ├── retriever.py             # Hybrid BM25 + dense retrieval
│   │   └── reranker.py              # Cross-encoder reranking
│   ├── candidate_model/
│   │   ├── model.py                 # Skill scores · gaps · STAR stories
│   │   └── updater.py               # Post-session update logic
│   ├── pipeline/
│   │   └── graph.py                 # LangGraph state machine + interview loop
│   ├── voice/
│   │   ├── client.py                # ElevenLabs streaming TTS + Scribe STT, retries
│   │   ├── cache.py                 # Content-addressed audio cache (atomic, LRU)
│   │   ├── interviewer.py           # Spoken questions/feedback, parallel prefetch
│   │   └── telemetry.py             # Time-to-first-audio p50/p95, cache hits, chars
│   ├── scheduler/
│   │   └── cron.py                  # APScheduler 5 AM cron job
│   ├── dashboard/
│   │   ├── app.py                   # Streamlit main app
│   │   ├── pages_job_tracker.py     # Job ranking + details view
│   │   ├── pages_skill_heatmap.py   # Candidate skill confidence scores
│   │   ├── pages_interview.py       # Interview Q&A history + scores
│   │   ├── pages_report.py          # Morning report display
│   │   ├── pages_controls.py        # In-browser controls: scrape · run pipeline · scheduler
│   │   └── pages_settings.py        # Configuration UI
│   ├── company_scraper.py           # Target-company career-page scraper (Playwright + ATS APIs)
│   ├── ats_fetchers.py              # Greenhouse/Lever/Ashby/Workable JSON-API clients
│   ├── etl.py                       # Resume PDF → text parsing
│   ├── jd_parser.py                 # Job description structuring
│   ├── llm_parser.py                # Local-first JD parser with Ollama fallback
│   ├── scoring.py                   # Local scoring engine
│   ├── job_store.py                 # Qdrant-backed job repository
│   ├── resume_tailor.py             # LLM-powered bullet rewriting
│   ├── cover_letter.py              # LLM cover letter generation
│   ├── pdf_generator.py             # LaTeX → PDF compilation
│   ├── job_imports.py               # CSV/JSON job loading
│   ├── qdrant_setup.py              # Qdrant client initialization
│   └── main.py                      # CLI entry point
├── data/
│   ├── master_resume.pdf            # Your resume (gitignored)
│   └── resume_template.tex          # LaTeX resume template
│   └── tests/
│       ├── test_scoring_agent.py
│       ├── test_cover_letter.py
│       ├── test_resume_tailor.py
│       ├── test_company_scraper.py
│       ├── test_w4.py
│       ├── test_w5.py
│       └── test_w6.py
├── outputs/
│   └── jobs/                        # Per-job: tailored resume + cover letter PDFs
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
├── .env.example
└── README.md
```

---

## Quickstart

> **Prerequisites:** Python 3.11+
> **Optional:** Docker (Postgres + Qdrant) · OpenAI API key · `pdflatex` (for PDF generation - `brew install --cask basictex` on macOS) · Ollama (local LLM)
>
> With zero setup you get: SQLite, local sentence-transformer scoring, no PDFs. With Docker + an OpenAI key you get the full LLM pipeline + Postgres + Qdrant.

```bash
# 1. Clone and install
git clone https://github.com/atharvahirulkar/JobPilot.git
cd JobPilot
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
playwright install chromium

# 2. Configure environment (sqlite + local scoring works out-of-the-box)
cp .env.example .env
# Optional: fill in OPENAI_API_KEY for LLM scoring + cover-letter quality

# 3. (Optional) Start infrastructure for full Postgres + Qdrant
docker compose up -d

# 4. Run the full pipeline once (scrapes 412 target companies → scores → tailors)
python -m jobpilot.main run_pipeline

# 5. Open the Streamlit dashboard
python -m jobpilot.main dashboard
# → http://localhost:8502

# 6. Schedule the 5 AM cron (Linux/macOS)
python -m jobpilot.main schedule --hour 5 --minute 0

# 7. Run a mock interview session
python -m jobpilot.main interview path/to/jd.json --questions 4

# 8. Check your skill confidence scores
python -m jobpilot.main skills
```

---

## Voice Mode (ElevenLabs)

Mock interviews can run as real spoken conversations. Set `ELEVENLABS_API_KEY`
in `.env` and toggle **Voice mode** on the Mock Interview page.

```
LangGraph generates questions
        │
        ▼
prefetch(): every question synthesized in parallel (thread pool)
        │      → content-addressed cache, so the session starts with zero wait
        ▼
Question plays (ElevenLabs Flash v2.5, streaming endpoint)
        │
        ▼
You answer out loud → ElevenLabs Scribe → editable transcript
        │
        ▼
LLM-as-judge scores it → short spoken verdict → next question
        │
        ▼
Session wrap-up spoken + telemetry panel (first-audio p50/p95, cache hit rate)
```

**Engineering details**

| Concern | How it's handled |
|---|---|
| Latency | Streaming TTS endpoint with time-to-first-audio measured per call; all questions prefetched concurrently before Q1 plays |
| Cost | Audio cached by `sha256(text, voice, model, format, settings)`, so repeated questions and replays bill zero characters |
| Reliability | Retries only on 429/5xx/network errors, exponential backoff with jitter, honors `Retry-After`; 4xx fails fast |
| Degradation | No key or any API failure → the session continues in text mode, never crashes |
| Concurrency safety | Atomic cache writes (temp file + `os.replace`), LRU eviction under a size cap |
| Observability | Every call (and cache hit) logged to `voice_events`; `jobpilot voice_stats` prints p50/p95 and usage |

```bash
python -m jobpilot.main voice_check   # synthesize a test line, report latency, prove the cache hit
python -m jobpilot.main voice_stats   # first-audio p50/p95, cache hit rate, chars billed, errors
```

Tests: `pytest src/jobpilot/tests/test_voice.py` runs against a mocked ElevenLabs API
(retries, fail-fast on 401, cache hits, concurrent prefetch, STT, telemetry), so no key is needed.

---

## The Story Behind This

I built JobPilot for myself. As an MS Data Science student actively job hunting, I was spending hours every day doing the same repetitive things - scanning job boards, copy-pasting resumes, rewriting cover letters, and still going into interviews underprepared.

So I automated it. Every layer of this system is something I use daily. The morning report, the tailored PDFs, the mock sessions - all of it runs while I sleep.

> *"The application you received was generated by it. Want to see the mock session it ran for this exact job description?"*

---

## Author

**Atharva Hirulkar** - MS Data Science, UC San Diego
[GitHub](https://github.com/atharvahirulkar) · [LinkedIn](https://linkedin.com/in/atharva-hirulkar)
