# 🤖 JobPilot

> **Your autonomous career agent.** Searches while you sleep. Preps while you wake.

![Status](https://img.shields.io/badge/status-in%20development-yellow)
![Python](https://img.shields.io/badge/python-3.11-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2-purple)
![AWS](https://img.shields.io/badge/AWS-ECS%20Fargate-orange)
![License](https://img.shields.io/badge/license-MIT-green)

---

## The Problem

Job searching is broken. You spend hours manually scanning portals, copy-pasting your resume, rewriting cover letters, and still walk into interviews unprepared for the specific role. It's slow, repetitive, and demoralizing.

**JobPilot automates all of it.**

---

## What It Does

Every morning at **5:00 AM**, JobPilot wakes up and runs a full pipeline — autonomously, while you sleep:

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
      Email digest + Streamlit dashboard — ready when you wake up
```

By the time you have your first coffee, you have a ranked shortlist of roles,
tailored application materials, and a mock interview session loaded for your top pick.

---

## Demo

> *Demo video and live dashboard screenshots coming W7 (Jun 2026)*

**Morning Report Preview:**
```
Good morning. Here's your June 10 job brief.

📊 Found 34 new jobs overnight.
🎯 6 strong matches (score > 75%)

Top Pick: Staff Data Scientist — Meta (AI Infra)
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
│    AWS EventBridge (5 AM cron)  │
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
│7 portals│ │0-100 + │ │PDF via         │
│Playwright│ │gaps    │ │WeasyPrint      │
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
| Agent framework | LangGraph — structured state machine with typed state |
| LLM | GPT-4o-mini (prod) · Claude Haiku (fallback) |
| Vector store | Qdrant — hybrid dense (all-MiniLM-L6-v2) + BM25 sparse |
| RAG evaluation | RAGAS — faithfulness · relevance · answer quality |
| Database | PostgreSQL — jobs · candidate model · answer history |
| ETL | PyPDF2 + spaCy + LLM extraction |
| Job scraping | Playwright + SerpAPI + RapidAPI |
| PDF generation | WeasyPrint |
| Observability | LangSmith — every agent node traced end-to-end |
| API | FastAPI |
| Frontend | Streamlit — job tracker · skill heatmap · mock interview UI |
| Scheduler | APScheduler + AWS EventBridge (5 AM cron) |
| Deployment | AWS ECS Fargate |
| CI/CD | GitHub Actions — push to main → ECR → ECS deploy |
| Language | Python 3.11 |

---

## Project Structure

```
jobpilot/
├── agents/
│   ├── job_search_agent.py      # Multi-portal scraper + LLM extractor
│   ├── scoring_agent.py         # Resume↔JD alignment + gap extraction
│   ├── resume_tailor.py         # Bullet rewriter + cover letter gen
│   ├── interview_agent.py       # Question gen + LLM-as-judge evaluator
│   └── report_agent.py          # Morning report + email sender
├── rag/
│   ├── ingest.py                # Resume + JD → Qdrant (hybrid index)
│   ├── retriever.py             # Hybrid BM25 + dense retrieval
│   └── reranker.py              # Cross-encoder reranking
├── candidate_model/
│   ├── model.py                 # Skill scores · gaps · STAR stories
│   └── updater.py               # Post-session update logic
├── pipeline/
│   └── graph.py                 # LangGraph state machine definition
├── scheduler/
│   └── cron.py                  # APScheduler 5 AM cron job
├── api/
│   └── main.py                  # FastAPI endpoints
├── dashboard/
│   └── app.py                   # Streamlit UI
├── outputs/
│   └── jobs/                    # Per-job: tailored resume + cover letter PDFs
├── data/
│   ├── master_resume.pdf        # Your resume (gitignored)
│   └── preferences.yaml         # Roles · locations · salary · avoid list
├── eval/
│   └── ragas_eval.py            # RAG + answer quality metrics
├── infra/
│   └── setup-ecs.sh             # One-time AWS ECS bootstrap
├── .github/
│   └── workflows/deploy.yml     # CI/CD — build → ECR → ECS
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quickstart

> **Prerequisites:** Python 3.11+ · Docker Desktop · OpenAI API key · SerpAPI key

```bash
# 1. Clone and install
git clone https://github.com/atharvahirulkar/jobpilot.git
cd jobpilot
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Fill in: OPENAI_API_KEY · SERPAPI_KEY · DATABASE_URL · QDRANT_URL

# 3. Start infrastructure
docker compose up -d
# PostgreSQL (5432) · Qdrant (6333) · MLflow (5001)

# 4. Ingest your resume
python rag/ingest.py --resume data/master_resume.pdf

# 5. Run the full pipeline manually
python pipeline/graph.py --run-now

# 6. Open the dashboard
streamlit run dashboard/app.py
# → http://localhost:8501

# 7. Schedule the 5 AM cron (local)
python scheduler/cron.py
```

---

## Build Log

| Week | Dates | Milestone | Status |
|---|---|---|---|
| W1 | May 1–7 | Resume ETL + Qdrant setup + JD parser | 🔄 In Progress |
| W2 | May 8–14 | Job search agent + PostgreSQL schema | ⬜ Upcoming |
| W3 | May 15–21 | Scoring engine + gap extractor + ranking | ⬜ Upcoming |
| W4 | May 22–28 | Resume tailor + cover letter + PDF output | ⬜ Upcoming |
| W5 | May 29–Jun 4 | Mock interview loop + evaluator + candidate model | ⬜ Upcoming |
| W6 | Jun 5–11 | Morning report + APScheduler + email digest | ⬜ Upcoming |
| W7 | Jun 12–18 | Streamlit dashboard + LangSmith + AWS deploy | ⬜ Upcoming |
| W8 | Jun 19–25 | Demo video + LinkedIn post + v1.0 release | ⬜ Upcoming |

---

## The Story Behind This

I built JobPilot for myself. As an MS Data Science student actively job hunting, I was spending hours every day doing the same repetitive things — scanning job boards, copy-pasting resumes, rewriting cover letters, and still going into interviews underprepared.

So I automated it. Every layer of this system is something I use daily. The morning report, the tailored PDFs, the mock sessions — all of it runs while I sleep.

> *"The cover letter you received was generated by it. Want to see the mock session it ran for this exact job description?"*

---

## Author

**Atharva Hirulkar** — MS Data Science, UC San Diego
[GitHub](https://github.com/atharvahirulkar) · [LinkedIn](https://linkedin.com/in/atharva-hirulkar)
