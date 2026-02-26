"""Tests for the evaluation module."""

import pytest

from src.evaluate import check_bias, check_data_integrity


class TestCheckBias:
    """Tests for bias detection in generated summaries."""

    def test_no_flags_on_identical_text(self):
        texts = ["The weather is sunny today."]
        flags = check_bias(texts, texts)
        assert flags == []

    def test_flags_overrepresented_term(self):
        sources = ["The person went to the store."] * 100
        # Summaries over-represent the word "she"
        summaries = ["She she she she she went shopping."] * 100
        flags = check_bias(sources, summaries, threshold=2.0)
        terms_flagged = [f["term"] for f in flags]
        assert "she" in terms_flagged


class TestCheckDataIntegrity:
    """Tests for data integrity checks."""

    def test_clean_data(self):
        records = [
            {"article": "Article A", "summary": "Summary A"},
            {"article": "Article B", "summary": "Summary B"},
        ]
        issues = check_data_integrity(records)
        assert issues["empty_article"] == 0
        assert issues["empty_summary"] == 0
        assert issues["duplicates"] == 0

    def test_detects_empty_article(self):
        records = [{"article": "", "summary": "Summary"}]
        issues = check_data_integrity(records)
        assert issues["empty_article"] == 1

    def test_detects_empty_summary(self):
        records = [{"article": "Article", "summary": ""}]
        issues = check_data_integrity(records)
        assert issues["empty_summary"] == 1

    def test_detects_duplicates(self):
        records = [
            {"article": "Same article", "summary": "Summary A"},
            {"article": "Same article", "summary": "Summary B"},
        ]
        issues = check_data_integrity(records)
        assert issues["duplicates"] == 1
