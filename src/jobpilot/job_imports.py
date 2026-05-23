from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from .job_normalize import dedupe_jobs, normalize_job


def load_jobs_file(path: str | Path) -> List[Dict[str, Any]]:
    """Load jobs from a JSON or CSV file.

    JSON accepts either a top-level list of job objects or a dict with a `jobs`
    key. CSV must contain headers such as title, company, location, description.
    """
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".json":
        return _load_jobs_json(file_path)
    if suffix == ".csv":
        return _load_jobs_csv(file_path)
    raise ValueError(f"Unsupported jobs file format: {file_path.suffix}")


def _load_jobs_json(file_path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(file_path.read_text())
    if isinstance(payload, list):
        return _normalize_and_dedupe([dict(item) for item in payload])
    if isinstance(payload, dict) and isinstance(payload.get("jobs"), list):
        return _normalize_and_dedupe([dict(item) for item in payload["jobs"]])
    raise ValueError("JSON jobs file must contain a list or a `jobs` key")


def _load_jobs_csv(file_path: Path) -> List[Dict[str, Any]]:
    with file_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return _normalize_and_dedupe([dict(row) for row in reader])


def _normalize_and_dedupe(jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = [normalize_job(job).to_dict() for job in jobs]
    return dedupe_jobs(normalized)
