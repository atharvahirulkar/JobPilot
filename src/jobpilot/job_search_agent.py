from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List

from tqdm import tqdm


@dataclass
class JobSearchResult:
    source: str
    jobs: List[Dict[str, Any]]
    error: str | None = None


class JobSearchPortal(ABC):
    """Abstract base for job search portals."""

    def __init__(self, portal_name: str):
        self.portal_name = portal_name

    @abstractmethod
    def search(self, query: str, location: str = "", limit: int = 20) -> JobSearchResult:
        """Search for jobs and return results."""
        pass


class LinkedInSearcher(JobSearchPortal):
    """LinkedIn job search via SerpAPI."""

    def __init__(self):
        super().__init__("LinkedIn")
        self.api_key = os.getenv("SERPAPI_API_KEY")

    def search(self, query: str, location: str = "", limit: int = 20) -> JobSearchResult:
        if not self.api_key:
            return JobSearchResult(self.portal_name, [], "SERPAPI_API_KEY not set")

        try:
            from serpapi import GoogleSearch
        except ImportError:
            return JobSearchResult(self.portal_name, [], "google-search-results not installed")

        try:
            params = {
                "engine": "google_jobs",
                "q": f"{query} {location}".strip(),
                "api_key": self.api_key,
            }
            search = GoogleSearch(params)
            results = search.get_dict()

            jobs = []
            for item in results.get("jobs_results", [])[:limit]:
                jobs.append(
                    {
                        "title": item.get("title", ""),
                        "company": item.get("company_name", ""),
                        "location": item.get("location", ""),
                        "description": item.get("description", ""),
                        "url": item.get("apply_link", ""),
                        "source": self.portal_name,
                    }
                )
            return JobSearchResult(self.portal_name, jobs)
        except Exception as e:
            return JobSearchResult(self.portal_name, [], str(e))


class IndeedSearcher(JobSearchPortal):
    """Indeed job search (placeholder for Playwright implementation)."""

    def __init__(self):
        super().__init__("Indeed")
        self.api_key = os.getenv("SERPAPI_API_KEY")

    def search(self, query: str, location: str = "", limit: int = 20) -> JobSearchResult:
        if not self.api_key:
            return JobSearchResult(self.portal_name, [], "SERPAPI_API_KEY not set")

        try:
            from serpapi import GoogleSearch
        except ImportError:
            return JobSearchResult(self.portal_name, [], "google-search-results not installed")

        try:
            params = {
                "engine": "google_jobs",
                "q": f"{query} site:indeed.com {location}".strip(),
                "api_key": self.api_key,
            }
            search = GoogleSearch(params)
            results = search.get_dict()

            jobs = []
            for item in results.get("jobs_results", [])[:limit]:
                jobs.append(
                    {
                        "title": item.get("title", ""),
                        "company": item.get("company_name", ""),
                        "location": item.get("location", ""),
                        "description": item.get("description", ""),
                        "url": item.get("apply_link", ""),
                        "source": self.portal_name,
                    }
                )
            return JobSearchResult(self.portal_name, jobs)
        except Exception as e:
            return JobSearchResult(self.portal_name, [], str(e))


class HandshakeSearcher(JobSearchPortal):
    """Handshake job search (placeholder)."""

    def __init__(self):
        super().__init__("Handshake")

    def search(self, query: str, location: str = "", limit: int = 20) -> JobSearchResult:
        return JobSearchResult(self.portal_name, [], "Handshake scraper not yet implemented")


class WellfoundSearcher(JobSearchPortal):
    """Wellfound job search (placeholder)."""

    def __init__(self):
        super().__init__("Wellfound")

    def search(self, query: str, location: str = "", limit: int = 20) -> JobSearchResult:
        return JobSearchResult(self.portal_name, [], "Wellfound scraper not yet implemented")


class JobSearchAgent:
    """Orchestrates job search across multiple portals."""

    def __init__(self):
        self.portals = [
            LinkedInSearcher(),
            IndeedSearcher(),
            HandshakeSearcher(),
            WellfoundSearcher(),
        ]

    def search(self, query: str = "Data Scientist", location: str = "Remote", limit_per_portal: int = 20) -> List[Dict[str, Any]]:
        """Search all portals and return aggregated results."""
        all_jobs = []

        for portal in tqdm(self.portals, desc="Searching job portals"):
            result = portal.search(query, location, limit_per_portal)
            if result.error:
                print(f"  ⚠ {result.source}: {result.error}")
            else:
                all_jobs.extend(result.jobs)
                print(f"  ✓ {result.source}: {len(result.jobs)} jobs")

        return all_jobs


if __name__ == "__main__":
    agent = JobSearchAgent()
    jobs = agent.search("Data Scientist", "Remote", limit_per_portal=5)
    print(f"\nTotal jobs found: {len(jobs)}")
    for job in jobs[:3]:
        print(job)
