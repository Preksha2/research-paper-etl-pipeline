"""Tests for data models."""
import pytest
from src.models.paper import Paper, Author, PipelineState


class TestPaper:

    def test_create_paper(self):
        paper = Paper(paper_id="abc", title="Test")
        assert paper.paper_id == "abc"
        assert paper.title == "Test"
        assert paper.confidence == 1.0
        assert paper.references == []

    def test_paper_with_authors(self):
        paper = Paper(
            paper_id="abc",
            title="Test",
            authors=[Author(name="Alice"), Author(name="Bob")],
        )
        assert len(paper.authors) == 2
        assert paper.authors[0].name == "Alice"


class TestPipelineState:

    def test_default_state(self):
        state = PipelineState()
        assert state.current_hop == 0
        assert state.max_hops == 2
        assert state.duplicate_count == 0
        assert state.papers_to_fetch == []
        assert state.all_processed_ids == []
