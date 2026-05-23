from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha1
from typing import Any, Dict, Iterable, List


@dataclass(frozen=True)
class NormalizedJob:
    job_id: str
    title: str
    company: str
    location: str
    description: str
    source: str = "manual"
    url: str = ""
    skills: List[str] | None = None
    responsibilities: List[str] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "title": self.title,
            "company": self.company,
            "location": self.location,
            "description": self.description,
            "source": self.source,
            "url": self.url,
            "skills": self.skills or [],
            "responsibilities": self.responsibilities or [],
        }


def normalize_job(job: Dict[str, Any]) -> NormalizedJob:
    data = {str(key).strip().lower(): value for key, value in dict(job).items()}

    title = _clean_text(_first_present(data, "title", "role", "position", "job_title"))
    company = _clean_text(_first_present(data, "company", "employer", "organization", "org"))
    location = _clean_text(_first_present(data, "location", "city", "work_location", "remote"))
    description = _clean_text(_first_present(data, "description", "jd", "job_description", "summary", "about"))
    source = _clean_text(_first_present(data, "source", "portal", "site")) or "manual"
    url = _clean_text(_first_present(data, "url", "job_url", "link", "apply_url"))

    skills = _parse_list(_first_present(data, "skills", "skill", "requirements"))
    responsibilities = _parse_list(_first_present(data, "responsibilities", "responsibility", "duties"))

    job_id = _clean_text(_first_present(data, "job_id", "id", "listing_id", "posting_id"))
    if not job_id:
        job_id = fingerprint_job(title, company, location, url, description)

    return NormalizedJob(
        job_id=job_id,
        title=title,
        company=company,
        location=location,
        description=description,
        source=source,
        url=url,
        skills=skills,
        responsibilities=responsibilities,
    )


def dedupe_jobs(jobs: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for job in jobs:
        normalized = normalize_job(job).to_dict()
        key = normalized["job_id"]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def fingerprint_job(title: str, company: str, location: str, url: str, description: str) -> str:
    fingerprint = "|".join(
        [
            _collapse_spaces(title).lower(),
            _collapse_spaces(company).lower(),
            _collapse_spaces(location).lower(),
            _collapse_spaces(url).lower(),
            _collapse_spaces(description).lower()[:500],
        ]
    )
    return sha1(fingerprint.encode("utf-8")).hexdigest()


def _first_present(data: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = data.get(key)
        if value not in (None, "", [], {}):
            return value
    return ""


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return _collapse_spaces(value)
    return _collapse_spaces(str(value))


def _collapse_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _parse_list(value: Any) -> List[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        items = re.split(r"[,;\n\u2022]|\s-\s", value)
    else:
        items = [value]
    return [
        _collapse_spaces(str(item))
        for item in items
        if _collapse_spaces(str(item))
    ]
