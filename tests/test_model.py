"""Tests for the model module."""

import pytest
from unittest.mock import patch, MagicMock

from src.model import compute_rouge


class TestComputeRouge:
    """Tests for the ROUGE metric computation."""

    def test_identical_strings_score_one(self):
        preds = ["The cat sat on the mat."]
        refs = ["The cat sat on the mat."]
        scores = compute_rouge(preds, refs)
        assert scores["rouge1"] == pytest.approx(1.0, abs=0.01)
        assert scores["rouge2"] == pytest.approx(1.0, abs=0.01)
        assert scores["rougeL"] == pytest.approx(1.0, abs=0.01)

    def test_completely_different_strings(self):
        preds = ["apple banana cherry"]
        refs = ["dog elephant fox"]
        scores = compute_rouge(preds, refs)
        assert scores["rouge1"] == pytest.approx(0.0, abs=0.01)

    def test_partial_overlap(self):
        preds = ["The quick brown fox"]
        refs = ["The slow brown dog"]
        scores = compute_rouge(preds, refs)
        # Expect non-zero but less than 1
        assert 0.0 < scores["rouge1"] < 1.0

    def test_multiple_pairs(self):
        preds = ["hello world", "foo bar"]
        refs = ["hello world", "baz qux"]
        scores = compute_rouge(preds, refs)
        # Average: first pair ≈1.0, second ≈0.0 → ~0.5
        assert 0.3 < scores["rouge1"] < 0.7
