"""W4 tests: LLMResumeTailor, LLMCoverLetterGenerator, PDFGenerator."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jobpilot.cover_letter import (
    CoverLetterRequest,
    LLMCoverLetterGenerator,
    build_request,
)
from jobpilot.pdf_generator import PDFGenerator, _cover_letter_tex, escape_latex
from jobpilot.resume_tailor import (
    LLMResumeTailor,
    _extract_resume_items,
    _extract_skills_block,
    _inject_bullets,
    _section_end,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TEMPLATE_PATH = Path(__file__).parent.parent.parent.parent / "data" / "resume_template.tex"

SAMPLE_JOB = {
    "title": "Machine Learning Engineer",
    "company": "Acme AI",
    "description": "Build ML pipelines, deploy models, work with Python, PyTorch, Kubernetes.",
    "skills": ["Python", "PyTorch", "Kubernetes", "ML pipelines", "FastAPI"],
}

MINI_TEX = r"""
\section{Education}
  \resumeItemListStart
    \resumeItem{Coursework: ML Systems, NLP}
  \resumeItemListEnd
\section{Projects}
  \resumeItemListStart
    \resumeItem{Built a fraud detection model using XGBoost.}
    \resumeItem{Deployed API on AWS ECS Fargate with CI/CD.}
  \resumeItemListEnd
\section{Experience}
  \resumeItemListStart
    \resumeItem{Led data pipeline automation across \textbf{50+ teams}.}
  \resumeItemListEnd
"""

MINI_TEX_WITH_SKILLS = r"""
\section{Technical Skills}
 \begin{itemize}[leftmargin=0.15in, label={}]
    \small{\item{
     \textbf{AI \& ML:} Python, PyTorch \\[2pt]
     \textbf{Cloud:} AWS, Docker
    }}
 \end{itemize}
"""


# ---------------------------------------------------------------------------
# _extract_resume_items
# ---------------------------------------------------------------------------

def test_extract_items_skips_education():
    items = _extract_resume_items(MINI_TEX)
    # Education coursework item must be excluded; 3 remaining items expected
    assert len(items) == 3
    contents = [c for _, _, c in items]
    assert all("Coursework" not in c for c in contents)


def test_extract_items_handles_nested_textbf():
    items = _extract_resume_items(MINI_TEX)
    contents = [c for _, _, c in items]
    assert any(r"\textbf{50+ teams}" in c for c in contents)


def test_extract_items_returns_correct_offsets():
    items = _extract_resume_items(MINI_TEX)
    for start, end, content in items:
        assert MINI_TEX[start:end] == content


# ---------------------------------------------------------------------------
# _inject_bullets
# ---------------------------------------------------------------------------

def test_inject_bullets_round_trips():
    items = _extract_resume_items(MINI_TEX)
    originals = [c for _, _, c in items]
    result = _inject_bullets(MINI_TEX, items, originals)
    assert result == MINI_TEX


def test_inject_bullets_replaces_content():
    items = _extract_resume_items(MINI_TEX)
    new_bullets = ["REWRITTEN_1", "REWRITTEN_2", "REWRITTEN_3"]
    result = _inject_bullets(MINI_TEX, items, new_bullets)
    assert "REWRITTEN_1" in result
    assert "REWRITTEN_2" in result
    assert "REWRITTEN_3" in result
    # Education item must be untouched
    assert "Coursework: ML Systems, NLP" in result


# ---------------------------------------------------------------------------
# _extract_skills_block
# ---------------------------------------------------------------------------

def test_extract_skills_block_finds_content():
    result = _extract_skills_block(MINI_TEX_WITH_SKILLS)
    assert result is not None
    _, _, content = result
    assert "PyTorch" in content
    assert r"\textbf{AI" in content


def test_extract_skills_block_returns_none_when_missing():
    assert _extract_skills_block(MINI_TEX) is None


# ---------------------------------------------------------------------------
# LLMResumeTailor — no API key fallback
# ---------------------------------------------------------------------------

def test_llm_resume_tailor_no_key_returns_unchanged():
    tailor = LLMResumeTailor(api_key="")
    result = tailor.tailor_latex(MINI_TEX, SAMPLE_JOB)
    assert result == MINI_TEX


def test_llm_resume_tailor_uses_template_on_exception():
    tailor = LLMResumeTailor(api_key="sk-fake")
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("api down")
    tailor._client = mock_client
    result = tailor.tailor_latex(MINI_TEX, SAMPLE_JOB)
    assert result == MINI_TEX


def test_llm_resume_tailor_injects_rewritten_bullets():
    tailor = LLMResumeTailor(api_key="sk-fake")

    rewritten = ["NEW_BULLET_A", "NEW_BULLET_B", "NEW_BULLET_C"]
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = json.dumps({"bullets": rewritten})

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_resp
    tailor._client = mock_client

    result = tailor.tailor_latex(MINI_TEX, SAMPLE_JOB)
    assert "NEW_BULLET_A" in result
    assert "NEW_BULLET_B" in result
    assert "Coursework: ML Systems, NLP" in result  # education untouched


def test_llm_resume_tailor_count_mismatch_falls_back():
    tailor = LLMResumeTailor(api_key="sk-fake")

    # LLM returns wrong number of bullets
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = json.dumps({"bullets": ["ONLY_ONE"]})

    mock_client = MagicMock()
    # First call = bullets, second call would be skills reorder
    mock_client.chat.completions.create.return_value = mock_resp
    tailor._client = mock_client

    result = tailor.tailor_latex(MINI_TEX, SAMPLE_JOB)
    # Should fall back to original when count mismatches
    assert "Built a fraud detection model" in result


def test_llm_resume_tailor_real_template_extracts_bullets():
    """Verify we can extract items from the actual resume template."""
    if not TEMPLATE_PATH.exists():
        pytest.skip("resume_template.tex not found")
    tex = TEMPLATE_PATH.read_text(encoding="utf-8")
    items = _extract_resume_items(tex)
    # Template: 3 project + 4 experience = 7 (education coursework item is skipped)
    assert len(items) == 7, f"Expected 7 non-education bullets, got {len(items)}"


# ---------------------------------------------------------------------------
# LLMCoverLetterGenerator — no API key fallback
# ---------------------------------------------------------------------------

REQUEST = CoverLetterRequest(
    candidate_name="Atharva Hirulkar",
    company="Acme AI",
    title="ML Engineer",
    resume_highlights=["Built XGBoost fraud model", "Deployed on AWS ECS"],
    skills=["Python", "PyTorch", "FastAPI"],
)


def test_llm_cover_letter_no_key_falls_back_to_template():
    gen = LLMCoverLetterGenerator(api_key="")
    result = gen.generate(REQUEST)
    assert "Acme AI" in result
    assert len(result) > 50


def test_llm_cover_letter_returns_llm_output():
    gen = LLMCoverLetterGenerator(api_key="sk-fake")
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = "Para1.\n\nPara2.\n\nPara3."
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_resp
    gen._client = mock_client

    result = gen.generate(REQUEST)
    assert result == "Para1.\n\nPara2.\n\nPara3."


def test_llm_cover_letter_falls_back_on_exception():
    gen = LLMCoverLetterGenerator(api_key="sk-fake")
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("timeout")
    gen._client = mock_client

    result = gen.generate(REQUEST)
    # Should fall back to template — contains company name
    assert "Acme AI" in result


def test_build_request_helper():
    req = build_request("Atharva Hirulkar", SAMPLE_JOB, ["bullet1", "bullet2"])
    assert req.company == "Acme AI"
    assert req.title == "Machine Learning Engineer"
    assert "Python" in req.skills


# ---------------------------------------------------------------------------
# PDFGenerator — unit tests (no actual pdflatex execution)
# ---------------------------------------------------------------------------

def test_escape_latex_special_chars():
    assert escape_latex("50% & $100") == r"50\% \& \$100"
    assert escape_latex("foo_bar") == r"foo\_bar"
    assert escape_latex("a#b") == r"a\#b"


def test_cover_letter_tex_contains_company_and_title():
    tex = _cover_letter_tex("Hello world.", {"company": "Acme AI", "title": "ML Engineer"})
    assert "Acme AI" in tex
    assert "ML Engineer" in tex
    assert r"\documentclass" in tex
    assert r"\end{document}" in tex


def test_cover_letter_tex_escapes_special_chars():
    tex = _cover_letter_tex("Hello.", {"company": "A&B Corp", "title": "Data_Scientist"})
    assert r"A\&B Corp" in tex
    assert r"Data\_Scientist" in tex


def test_pdf_generator_raises_if_no_pdflatex():
    gen = PDFGenerator()
    with patch("shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="pdflatex not found"):
            gen.compile_latex(r"\documentclass{article}\begin{document}hi\end{document}", Path("/tmp/test.pdf"))


def test_pdf_generator_output_paths(tmp_path):
    gen = PDFGenerator(outputs_root=tmp_path)
    # Mock compile_latex so no actual pdflatex is needed
    gen.compile_latex = lambda tex, path: path
    resume_path = gen.resume_pdf("dummy tex", "acme_mle")
    cl_path     = gen.cover_letter_pdf("Body text.", {"company": "Acme", "title": "MLE"}, "acme_mle")
    assert resume_path == tmp_path / "acme_mle" / "tailored_resume.pdf"
    assert cl_path     == tmp_path / "acme_mle" / "cover_letter.pdf"
