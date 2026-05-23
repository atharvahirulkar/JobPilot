# 🤖 JobPilot

> **Your autonomous career agent.** Searches while you sleep. Preps while you wake.

![Status](https://img.shields.io/badge/status-in%20development-yellow)
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

Every morning at **5:00 AM**, JobPilot wakes up and runs a full pipeline - autonomously, while you sleep:

```
🔍  Searches 7 job portals globally for new DS/MLE roles
      LinkedIn · Indeed · Handshake · Wellfound · Greenhouse · Lever · Google Jobs

📊  Scores every JD against your resume using an LLM alignment agent
      Match score 0–100 · Skill gaps · Role fit · Company tier

✍️  For top matches (>75%), tailors your resume + writes a cover letter
      Reorders bullets by relevance · Injects keywords naturally
      Outputs: tailored_resume.pdf + cover_letter.pdf per job

📋  Compiles your morning report
      Top 5 jobs · Materials ready · Gaps flagged · Prep status

🎯  Auto-queues a personalized mock interview session
      Questions weighted by YOUR weakest skill areas for today's top job

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
# → http://localhost:8501
```

**Morning Report Example:**
```
Good morning. Here's your June 10 job brief.

📊 Found 34 new jobs overnight.
🎯 6 strong matches (score > 75%)

Top Pick: Staff Data Scientist - Meta (AI Infra)
  Match Score: 91% | Gap: distributed systems
  📄 Tailored resume → outputs/jobs/meta_ai_infra/tailored_resume.pdf
  📝 Cover letter   → outputs/jobs/meta_ai_infra/cover_letter.pdf

🎤 Mock interview session queued.
   Focus: distributed systems · ML system design · experimentation
   Open dashboard → http://localhost:8501
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
│7 portals│ │0-100 + │ │LaTeX → PDF     │
│Playwright│ │gaps    │ │via WeasyPrint  │
│+SerpAPI │ │        │ │                │
└────┬────┘ └───┬────┘ └───────┬────────┘
     │          │              │
┌────▼──────────▼──────────────▼────────┐
│             PostgreSQL                │
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
| Job scraping | Playwright + SerpAPI + RapidAPI |
| PDF generation | WeasyPrint |
| Observability | LangSmith - every agent node traced end-to-end |
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
│   ├── scheduler/
│   │   └── cron.py                  # APScheduler 5 AM cron job
│   ├── dashboard/
│   │   ├── app.py                   # Streamlit main app
│   │   ├── pages_job_tracker.py     # Job ranking + details view
│   │   ├── pages_skill_heatmap.py   # Candidate skill confidence scores
│   │   ├── pages_interview.py       # Interview Q&A history + scores
│   │   ├── pages_report.py          # Morning report display
│   │   └── pages_settings.py        # Configuration UI
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
├── tests/
│   ├── test_scoring_agent.py
│   ├── test_cover_letter.py
│   ├── test_resume_tailor.py
│   ├── test_w4.py
│   ├── test_w5.py
│   └── test_w6.py
├── outputs/
│   └── jobs/                        # Per-job: tailored resume + cover letter PDFs
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quickstart

> **Prerequisites:** Python 3.11+ · PostgreSQL + Qdrant (Docker) · OpenAI API key

```bash
# 1. Clone and install
git clone https://github.com/atharvahirulkar/JobPilot.git
cd JobPilot
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Configure environment
cp .env.example .env
# Fill in: OPENAI_API_KEY · SERPAPI_KEY · DATABASE_URL · QDRANT_URL

# 3. Start infrastructure
docker compose up -d
# PostgreSQL (5432) · Qdrant (6333)

# 4. Run the full pipeline once
python -m jobpilot.main run_pipeline

# 5. Open the Streamlit dashboard
python -m jobpilot.main dashboard
# → http://localhost:8501

# 6. Schedule the 5 AM cron (Linux/macOS)
python -m jobpilot.main schedule --hour 5 --minute 0

# 7. Run a mock interview session
python -m jobpilot.main interview path/to/jd.json --questions 4

# 8. Check your skill confidence scores
python -m jobpilot.main skills
```

## The Story Behind This

I built JobPilot for myself. As an MS Data Science student actively job hunting, I was spending hours every day doing the same repetitive things - scanning job boards, copy-pasting resumes, rewriting cover letters, and still going into interviews underprepared.

So I automated it. Every layer of this system is something I use daily. The morning report, the tailored PDFs, the mock sessions - all of it runs while I sleep.

> *"The application you received was generated by it. Want to see the mock session it ran for this exact job description?"*

---

## Author

**Atharva Hirulkar** - MS Data Science, UC San Diego
[GitHub](https://github.com/atharvahirulkar) · [LinkedIn](https://linkedin.com/in/atharva-hirulkar)
