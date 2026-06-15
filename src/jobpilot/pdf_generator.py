from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict

_LATEX_ESCAPE = str.maketrans({
    "&":  r"\&",
    "%":  r"\%",
    "$":  r"\$",
    "#":  r"\#",
    "_":  r"\_",
    "{":  r"\{",
    "}":  r"\}",
    "~":  r"\textasciitilde{}",
    "^":  r"\textasciicircum{}",
    "\\": r"\textbackslash{}",
})


def escape_latex(text: str) -> str:
    return str(text).translate(_LATEX_ESCAPE)


def _cover_letter_tex(
    body: str,
    job: Dict[str, Any],
    candidate_name: str = "[Your Name]",
    candidate_email: str = "",
    candidate_phone: str = "",
    candidate_linkedin: str = "",
    candidate_portfolio: str = "",
) -> str:
    """Wrap plain-text cover letter body in a LaTeX document matching resume style."""
    company = escape_latex(str(job.get("company", "the company")))
    title   = escape_latex(str(job.get("title",   "the role")))

    # Build contact line — only include non-empty fields
    contact_parts = []
    if candidate_email:
        contact_parts.append(escape_latex(candidate_email))
    if candidate_phone:
        contact_parts.append(escape_latex(candidate_phone))
    if candidate_linkedin:
        # Render as clickable hyperlink
        url = candidate_linkedin if candidate_linkedin.startswith("http") else f"https://{candidate_linkedin}"
        contact_parts.append(rf"\href{{{escape_latex(url)}}}{{{escape_latex(candidate_linkedin)}}}")
    if candidate_portfolio:
        url = candidate_portfolio if candidate_portfolio.startswith("http") else f"https://{candidate_portfolio}"
        contact_parts.append(rf"\href{{{escape_latex(url)}}}{{{escape_latex(candidate_portfolio)}}}")

    contact_line = r" \quad $|$ \quad ".join(contact_parts) if contact_parts else ""

    # Convert paragraphs: blank lines → \medskip\noindent
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", body.strip()) if p.strip()]
    body_tex = "\n\n\\medskip\n\\noindent ".join(escape_latex(p) for p in paragraphs)

    header_block = rf"\noindent\textbf{{\Large {escape_latex(candidate_name)}}}\\[4pt]"
    if contact_line:
        header_block += f"\n\\noindent {contact_line}\\\\[8pt]"
    else:
        header_block += "\n\\vspace{4pt}"

    return rf"""\documentclass[letterpaper,11pt]{{article}}

\usepackage[empty]{{fullpage}}
\usepackage[hidelinks]{{hyperref}}
\usepackage[english]{{babel}}
\usepackage{{geometry}}
\geometry{{letterpaper, top=1in, bottom=1in, left=1in, right=1in}}

\begin{{document}}

{header_block}

\noindent Hiring Team\\
\noindent {company}\\[12pt]

\noindent\textbf{{Re: {title}}}\\[12pt]

\noindent {body_tex}

\vspace{{16pt}}
\noindent Sincerely,\\[4pt]
\noindent\textbf{{{escape_latex(candidate_name)}}}

\end{{document}}
"""


class PDFGenerator:
    """Compile LaTeX strings to PDF using pdflatex.

    Raises RuntimeError if pdflatex is not on PATH or compilation fails.
    """

    def __init__(self, outputs_root: str | Path = "outputs/jobs"):
        self.outputs_root = Path(outputs_root)

    def compile_latex(self, tex_string: str, output_path: Path) -> Path:
        """Write tex_string to a temp dir, compile twice (for refs), copy PDF to output_path."""
        pdflatex = shutil.which("pdflatex")
        if not pdflatex:
            raise RuntimeError(
                "pdflatex not found on PATH. Install MacTeX: brew install --cask mactex-no-gui"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            tex_file = tmp / "document.tex"
            tex_file.write_text(tex_string, encoding="utf-8")

            cmd = [
                pdflatex,
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-output-directory", str(tmp),
                str(tex_file),
            ]
            for _ in range(2):  # two passes for correct cross-references
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(tmp))

            pdf_tmp = tmp / "document.pdf"
            if not pdf_tmp.exists():
                log_tail = result.stdout[-3000:] + result.stderr[-500:]
                raise RuntimeError(f"pdflatex compilation failed:\n{log_tail}")

            shutil.copy(pdf_tmp, output_path)

        return output_path

    def resume_pdf(self, tailored_tex: str, job_id: str) -> Path:
        """Compile tailored LaTeX → outputs/jobs/{job_id}/tailored_resume.pdf"""
        out = self.outputs_root / job_id / "tailored_resume.pdf"
        return self.compile_latex(tailored_tex, out)

    def cover_letter_pdf(
        self,
        cover_letter_text: str,
        job: Dict[str, Any],
        job_id: str,
        candidate_name: str = "",
        candidate_email: str = "",
        candidate_phone: str = "",
        candidate_linkedin: str = "",
        candidate_portfolio: str = "",
    ) -> Path:
        """Compile plain-text cover letter → outputs/jobs/{job_id}/cover_letter.pdf"""
        tex = _cover_letter_tex(
            cover_letter_text, job,
            candidate_name=candidate_name or "[Your Name]",
            candidate_email=candidate_email,
            candidate_phone=candidate_phone,
            candidate_linkedin=candidate_linkedin,
            candidate_portfolio=candidate_portfolio,
        )
        out = self.outputs_root / job_id / "cover_letter.pdf"
        return self.compile_latex(tex, out)
