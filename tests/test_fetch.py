"""Tests for the fetch node."""
import pytest
from src.nodes.fetch import extract_paper_id


class TestExtractPaperId:

    def test_arxiv_url(self):
        url = "https://arxiv.org/abs/1706.03762"
        assert extract_paper_id(url) == "ArXiv:1706.03762"

    def test_doi_string(self):
        doi = "10.1038/nature12373"
        assert extract_paper_id(doi) == "DOI:10.1038/nature12373"

    def test_doi_url(self):
        url = "https://doi.org/10.1038/nature12373"
        assert extract_paper_id(url) == "DOI:10.1038/nature12373"

    def test_semantic_scholar_hash(self):
        sid = "a" * 40
        assert extract_paper_id(sid) == "a" * 40

    def test_plain_id_passthrough(self):
        pid = "some-paper-id"
        assert extract_paper_id(pid) == "some-paper-id"
