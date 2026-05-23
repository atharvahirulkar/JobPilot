from typing import Dict, Any, List
import re
import spacy


class JDParser:
    """Minimal JD parser for extracting title, skills, and description.

    Heuristics here are intentionally simple for W1 and will be improved
    in later iterations (NER, regex lists, LLM extraction).
    """

    def __init__(self, nlp=None):
        self.nlp = nlp or spacy.load("en_core_web_sm")

    def parse(self, text: str) -> Dict[str, Any]:
        title = self._extract_title(text)
        skills = self._extract_skills(text)
        description = text.strip()
        return {"title": title, "skills": skills, "description": description}

    def _extract_title(self, text: str) -> str:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        return lines[0] if lines else ""

    def _extract_skills(self, text: str) -> List[str]:
        m = re.search(r"(Skills|Requirements|Qualifications)[:\\s\\-]*(.*)", text, re.IGNORECASE | re.DOTALL)
        if m:
            skills_blob = m.group(2)
            parts = re.split(r",|;|\\n", skills_blob)
            return [p.strip() for p in parts if p.strip()][:40]
        # fallback: noun chunks
        doc = self.nlp(text)
        candidates = [chunk.text for chunk in doc.noun_chunks if len(chunk.text) < 60]
        return candidates[:40]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Demo JD Parser")
    parser.add_argument("jdfile", help="Path to text file containing JD")
    args = parser.parse_args()
    text = open(args.jdfile, "r").read()
    parsed = JDParser().parse(text)
    print(parsed)
