"""Tests for the extract node."""
import pytest
from src.models.paper import Paper, Author
from src.nodes.extract import _heuristic_enrich


class TestHeuristicEnrich:

    def test_full_metadata(self):
        paper = Paper(
            paper_id="test",
            title="Full Paper",
            authors=[Author(name="Alice")],
            year=2023,
            abstract="A" * 200,
            doi="10.1234/test",
        )
        enriched = _heuristic_enrich(paper)
        assert enriched.confidence == 1.0

    def test_minimal_metadata(self):
        paper = Paper(
            paper_id="test",
            title="Bare Paper",
            authors=[],
        )
        enriched = _heuristic_enrich(paper)
        assert enriched.confidence == 0.5

    def test_partial_metadata(self):
        paper = Paper(
            paper_id="test",
            title="Some Paper",
            authors=[Author(name="Bob")],
            year=2022,
        )
        enriched = _heuristic_enrich(paper)
        assert 0.6 <= enriched.confidence <= 0.8

    def test_sets_extraction_notes(self):
        paper = Paper(paper_id="test", title="Test")
        enriched = _heuristic_enrich(paper)
        assert "heuristic" in enriched.extraction_notes.lower()
