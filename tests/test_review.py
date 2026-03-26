"""Tests for the review node."""
import pytest
from src.models.paper import Paper, Author
from src.nodes.review import auto_fix_paper


class TestAutoFix:

    def test_extract_year_from_title(self):
        paper = Paper(
            paper_id="test",
            title="Deep Learning Advances (2023)",
            confidence=0.5,
        )
        fixed = auto_fix_paper(paper)
        assert fixed.year == 2023
        assert fixed.confidence > 0.5

    def test_no_year_in_title(self):
        paper = Paper(
            paper_id="test",
            title="Some Paper About AI",
            confidence=0.5,
        )
        fixed = auto_fix_paper(paper)
        assert fixed.year is None

    def test_short_abstract_flagged(self):
        paper = Paper(
            paper_id="test",
            title="Short Abstract Paper",
            abstract="Too short",
            confidence=0.8,
        )
        fixed = auto_fix_paper(paper)
        assert "short" in (fixed.extraction_notes or "").lower()
