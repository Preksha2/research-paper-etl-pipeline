"""Tests for the parse node."""
import pytest
from src.nodes.parse import parse_raw_paper


class TestParseRawPaper:

    def test_basic_parse(self):
        raw = {
            "paperId": "abc123",
            "title": "Test Paper",
            "authors": [{"name": "Alice", "authorId": "1"}, {"name": "Bob", "authorId": "2"}],
            "year": 2023,
            "abstract": "This is a test abstract.",
            "externalIds": {"DOI": "10.1234/test", "ArXiv": "2301.00001"},
            "references": [{"paperId": "ref1"}, {"paperId": "ref2"}],
            "citationCount": 100,
            "url": "https://example.com",
        }
        paper = parse_raw_paper(raw)

        assert paper.paper_id == "abc123"
        assert paper.title == "Test Paper"
        assert len(paper.authors) == 2
        assert paper.authors[0].name == "Alice"
        assert paper.year == 2023
        assert paper.doi == "10.1234/test"
        assert paper.arxiv_id == "2301.00001"
        assert len(paper.references) == 2
        assert paper.citation_count == 100

    def test_missing_fields(self):
        raw = {
            "paperId": "xyz",
            "title": "Minimal Paper",
            "authors": [],
            "externalIds": {},
            "references": [],
        }
        paper = parse_raw_paper(raw)

        assert paper.paper_id == "xyz"
        assert paper.title == "Minimal Paper"
        assert paper.year is None
        assert paper.doi is None
        assert paper.abstract is None
        assert len(paper.references) == 0

    def test_null_references(self):
        raw = {
            "paperId": "test",
            "title": "No Refs",
            "authors": [],
            "externalIds": None,
            "references": None,
        }
        paper = parse_raw_paper(raw)
        assert len(paper.references) == 0

    def test_confidence_with_doi(self):
        raw = {
            "paperId": "test",
            "title": "Has DOI",
            "authors": [],
            "externalIds": {"DOI": "10.1234/x"},
            "references": [],
        }
        paper = parse_raw_paper(raw)
        assert paper.confidence == 1.0

    def test_confidence_without_doi(self):
        raw = {
            "paperId": "test",
            "title": "No DOI",
            "authors": [],
            "externalIds": {},
            "references": [],
        }
        paper = parse_raw_paper(raw)
        assert paper.confidence == 0.8
