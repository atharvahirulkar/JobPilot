from pathlib import Path
from typing import Dict, Any

from PyPDF2 import PdfReader
import spacy


class ResumeETL:
    """Simple resume ETL: PDF -> text -> spaCy parse

    This is a minimal, extensible skeleton for Week 1.
    """

    def __init__(self, nlp=None):
        self.nlp = nlp or spacy.load("en_core_web_sm")

    def extract_text_from_pdf(self, path: str) -> str:
        path = Path(path)
        reader = PdfReader(str(path))
        texts = []
        for page in reader.pages:
            texts.append(page.extract_text() or "")
        return "\n".join(texts)

    def parse_resume(self, text: str) -> Dict[str, Any]:
        doc = self.nlp(text)
        orgs = list({ent.text for ent in doc.ents if ent.label_ in ("ORG",)})
        people = list({ent.text for ent in doc.ents if ent.label_ in ("PERSON",)})
        # lightweight skill candidate extraction via noun chunks
        skills = [chunk.text for chunk in doc.noun_chunks][:50]
        return {"text": text, "orgs": orgs, "people": people, "skills": skills}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Demo Resume ETL")
    parser.add_argument("pdf", help="Path to resume PDF")
    args = parser.parse_args()
    etl = ResumeETL()
    txt = etl.extract_text_from_pdf(args.pdf)
    parsed = etl.parse_resume(txt)
    print(parsed)
