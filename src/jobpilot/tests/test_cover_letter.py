"""Unit tests for cover_letter.py"""
import pytest
from jobpilot.cover_letter import CoverLetterGenerator, CoverLetterRequest, build_request


def test_cover_letter_generator_basic():
    """Test basic cover letter generation."""
    gen = CoverLetterGenerator()
    request = CoverLetterRequest(
        candidate_name="John Doe",
        company="Acme Corp",
        title="Data Scientist",
        resume_highlights=["Led ML project to production", "Improved model accuracy by 15%"],
        skills=["Python", "SQL", "Machine Learning"]
    )
    
    result = gen.generate(request)
    
    assert isinstance(result, str)
    assert "John Doe" in result
    assert "Acme Corp" in result
    assert "Data Scientist" in result
    assert len(result) > 50  # Should have substantial content


def test_cover_letter_format():
    """Test cover letter has proper structure."""
    gen = CoverLetterGenerator()
    request = CoverLetterRequest(
        candidate_name="Jane Smith",
        company="Tech Startup",
        title="ML Engineer",
        resume_highlights=["Built recommendation engine"],
        skills=["Python", "TensorFlow"]
    )
    
    result = gen.generate(request)
    
    # Should have intro, body, and closing
    lines = result.split("\n")
    assert len(lines) > 3
    
    # Should have salutation
    assert "Dear" in result
    
    # Should have candidate name at end
    assert "Jane Smith" in result.split("\n")[-1]
    
    # Should have "Sincerely"
    assert "Sincerely" in result


def test_cover_letter_skill_formatting():
    """Test skills are formatted correctly."""
    gen = CoverLetterGenerator()
    request = CoverLetterRequest(
        candidate_name="Alice",
        company="Company",
        title="Engineer",
        resume_highlights=["Did work"],
        skills=["Python", "Docker", "Kubernetes", "AWS", "GCP"]
    )
    
    result = gen.generate(request)
    
    # Should have first 5 skills formatted
    assert "Python" in result
    assert "Docker" in result


def test_cover_letter_empty_highlights():
    """Test cover letter handles empty highlights gracefully."""
    gen = CoverLetterGenerator()
    request = CoverLetterRequest(
        candidate_name="Bob",
        company="Company",
        title="Role",
        resume_highlights=[],
        skills=["Skill1"]
    )
    
    result = gen.generate(request)
    
    # Should still generate valid letter
    assert "Bob" in result
    assert "Company" in result
    assert len(result) > 50


def test_cover_letter_empty_skills():
    """Test cover letter handles empty skills gracefully."""
    gen = CoverLetterGenerator()
    request = CoverLetterRequest(
        candidate_name="Charlie",
        company="Company",
        title="Role",
        resume_highlights=["Achievement 1"],
        skills=[]
    )
    
    result = gen.generate(request)
    
    # Should still generate valid letter with default skills message
    assert "Charlie" in result
    assert "Company" in result


def test_build_request_helper():
    """Test build_request helper."""
    candidate = "Test Candidate"
    job = {
        "job_id": "j1",
        "title": "Senior Role",
        "company": "Great Corp",
        "skills": ["Python", "SQL"]
    }
    bullets = ["Achievement 1", "Achievement 2"]
    
    request = build_request(candidate, job, bullets)
    
    assert request.candidate_name == "Test Candidate"
    assert request.company == "Great Corp"
    assert request.title == "Senior Role"
    assert request.resume_highlights == bullets


def test_cover_letter_no_double_periods():
    """Test cover letter doesn't have double periods from sentence cleaning."""
    gen = CoverLetterGenerator()
    request = CoverLetterRequest(
        candidate_name="David",
        company="Company",
        title="Role",
        resume_highlights=["Did something."],  # Bullet has period
        skills=["Python"]
    )
    
    result = gen.generate(request)
    
    # Should not have ".." patterns
    assert ".." not in result


def test_cover_letter_deterministic():
    """Test cover letter generation is deterministic."""
    gen = CoverLetterGenerator()
    request = CoverLetterRequest(
        candidate_name="Eve",
        company="Company",
        title="Role",
        resume_highlights=["Highlight"],
        skills=["Skill"]
    )
    
    result1 = gen.generate(request)
    result2 = gen.generate(request)
    
    # Same input should produce same output
    assert result1 == result2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
