"""Tests for the data preprocessing module."""

import json
import os
import tempfile

import pytest

from src.data_preprocess import clean_text, build_dataset_from_jsonl


class TestCleanText:
    """Tests for the clean_text function."""

    def test_collapses_whitespace(self):
        assert clean_text("hello   world") == "hello world"

    def test_collapses_newlines(self):
        assert clean_text("hello\n\n\nworld") == "hello world"

    def test_strips_leading_trailing(self):
        assert clean_text("  hello world  ") == "hello world"

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_preserves_normal_text(self):
        text = "This is a normal sentence."
        assert clean_text(text) == text


class TestBuildDatasetFromJsonl:
    """Tests for loading JSONL datasets."""

    def _write_jsonl(self, records, path):
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    def test_loads_valid_records(self, tmp_path):
        records = [
            {"article": "Article one.", "summary": "Summary one."},
            {"article": "Article two.", "summary": "Summary two."},
        ]
        path = str(tmp_path / "data.jsonl")
        self._write_jsonl(records, path)
        loaded = build_dataset_from_jsonl(path)
        assert len(loaded) == 2
        assert loaded[0]["article"] == "Article one."

    def test_skips_empty_lines(self, tmp_path):
        path = str(tmp_path / "data.jsonl")
        with open(path, "w") as f:
            f.write('{"article": "A", "summary": "S"}\n\n\n')
        loaded = build_dataset_from_jsonl(path)
        assert len(loaded) == 1

    def test_skips_missing_keys(self, tmp_path):
        records = [{"title": "no article key"}]
        path = str(tmp_path / "data.jsonl")
        self._write_jsonl(records, path)
        loaded = build_dataset_from_jsonl(path)
        assert len(loaded) == 0

    def test_skips_invalid_json(self, tmp_path):
        path = str(tmp_path / "data.jsonl")
        with open(path, "w") as f:
            f.write("not json\n")
        loaded = build_dataset_from_jsonl(path)
        assert len(loaded) == 0
