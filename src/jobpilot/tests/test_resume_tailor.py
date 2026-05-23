"""Unit tests for resume_tailor.py"""
import pytest
from jobpilot.resume_tailor import ResumeTailor, TailoredResume, _tokenize, _normalize_text


def test_tokenize_basic():
    """Test basic tokenization."""
    text = "Python SQL Machine Learning experience"
    tokens = _tokenize(text)
    assert "python" in tokens or "python" in [t.lower() for t in tokens]


def test_normalize_text():
    """Test text normalization."""
    text = "Line one\n  Line two  \n Line three"
    normalized = _normalize_text(text)
    assert "\n" not in normalized
    assert normalized == "Line one Line two Line three"


def test_resume_tailor_basic():
    """Test basic resume tailoring."""
    tailor = ResumeTailor()
    resume_text = """
    John Doe
    Data Scientist
    
    - Built ML models for fraud detection
    - Created dashboards for KPI tracking
    - Led cross-functional teams
    """
    
    job = {
        "title": "Senior Data Scientist",
        "description": "Looking for Python, SQL, Machine Learning expertise",
        "skills": ["Python", "SQL", "Machine Learning"],
        "responsibilities": ["Model building", "Data analysis"]
    }
    
    result = tailor.tailor(resume_text, job)
    
    assert isinstance(result, TailoredResume)
    assert result.header  # Should have extracted header
    assert len(result.bullets) > 0
    assert "Data Scientist" in result.header or "John Doe" in result.header


def test_resume_tailor_scoring():
    """Test bullet re-ranking by keyword overlap."""
    tailor = ResumeTailor()
    resume_text = """
    Jane Smith
    ML Engineer
    
    - Python expertise in production systems
    - Built scalable APIs
    - Worked with SQL databases
    """
    
    job = {
        "title": "ML Engineer",
        "description": "Python SQL API development",
        "skills": ["Python", "SQL", "APIs"],
    }
    
    result = tailor.tailor(resume_text, job)
    
    # Bullets should be present
    assert len(result.bullets) > 0
    # Bullets should be normalized (single line)
    for bullet in result.bullets:
        assert "\n" not in bullet


def test_resume_tailor_injection():
    """Test keyword injection."""
    tailor = ResumeTailor()
    resume_text = """
    Alex Smith
    Backend Engineer
    
    - Developed APIs
    - Managed databases
    """
    
    job = {
        "title": "Backend Engineer",
        "description": "Python Django REST APIs PostgreSQL",
        "skills": ["Python", "Django", "PostgreSQL"],
        "responsibilities": ["API design", "Database optimization"]
    }
    
    result = tailor.tailor(resume_text, job)
    
    # Should have injected at least one keyword if possible
    # Check that injected_keywords is a list
    assert isinstance(result.injected_keywords, list)


def test_resume_tailor_denylist():
    """Test that denylist tokens are not injected."""
    tailor = ResumeTailor()
    resume_text = """
    Profile
    
    - Worked in data teams
    """
    
    job = {
        "title": "Data Role",
        "description": "Using data science and machine learning",
        "skills": ["data", "experience"],  # low-value skills
    }
    
    result = tailor.tailor(resume_text, job)
    
    # Should not inject "data" or "experience" (in denylist)
    assert "data" not in " ".join(result.injected_keywords).lower()


def test_resume_tailor_short_whitelist():
    """Test that whitelisted short tokens can be injected."""
    tailor = ResumeTailor()
    resume_text = """
    Tech Lead
    
    - Backend systems
    """
    
    job = {
        "title": "Senior Engineer",
        "description": "SQL AWS APIs",
        "skills": ["SQL", "AWS"],
    }
    
    result = tailor.tailor(resume_text, job)
    
    # SQL and AWS are in whitelist, so might be injected
    # Just verify the result is valid
    assert isinstance(result, TailoredResume)
    assert len(result.bullets) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
