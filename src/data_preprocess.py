"""
Data Preprocessing Module for the Article Summarization System.

Handles loading raw articles from various formats (TXT, PDF, HTML),
cleaning and normalizing text, tokenizing with HuggingFace tokenizers,
and exporting processed examples as JSONL files.
"""

import os
import re
import json
import glob
import argparse
from typing import List, Dict, Optional, Tuple

from tqdm import tqdm
from transformers import AutoTokenizer

from src.utils import setup_logging, ensure_dir

logger = setup_logging()

# ---------------------------------------------------------------------------
# Readers – each returns the raw text from a single file
# ---------------------------------------------------------------------------

def read_txt(filepath: str) -> str:
    """Read a plain-text file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def read_html(filepath: str) -> str:
    """Extract visible text from an HTML file using BeautifulSoup."""
    from bs4 import BeautifulSoup

    with open(filepath, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "lxml")

    # Remove scripts, styles, and navigation elements
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    return soup.get_text(separator="\n")


def read_pdf(filepath: str) -> str:
    """Extract text from a PDF using pdfminer.six."""
    from pdfminer.high_level import extract_text

    return extract_text(filepath)


READERS = {
    ".txt": read_txt,
    ".html": read_html,
    ".htm": read_html,
    ".pdf": read_pdf,
}

# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Clean and normalise raw article text.

    Steps:
    1. Replace multiple whitespace / newlines with a single space.
    2. Strip leading/trailing whitespace.
    3. Remove non-printable characters.
    """
    text = re.sub(r"\s+", " ", text)           # collapse whitespace
    text = re.sub(r"[^\x20-\x7E\n]", "", text) # keep printable ASCII + newline
    return text.strip()


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

def tokenize_pair(
    article: str,
    summary: str,
    tokenizer: AutoTokenizer,
    max_input_length: int = 1024,
    max_target_length: int = 256,
) -> Dict:
    """
    Tokenize an (article, summary) pair for a seq2seq model.

    Returns a dict ready for HuggingFace `Dataset.from_dict`.
    """
    model_inputs = tokenizer(
        article,
        max_length=max_input_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    labels = tokenizer(
        summary,
        max_length=max_target_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    model_inputs["labels"] = labels["input_ids"]
    return {k: v.squeeze(0).tolist() for k, v in model_inputs.items()}


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def load_raw_articles(input_dir: str) -> List[Tuple[str, str]]:
    """
    Walk *input_dir* and return a list of (filepath, raw_text) tuples.

    Recognises .txt, .html, .htm, and .pdf files.
    """
    articles: List[Tuple[str, str]] = []
    for ext, reader in READERS.items():
        for path in glob.glob(os.path.join(input_dir, f"**/*{ext}"), recursive=True):
            try:
                text = reader(path)
                articles.append((path, text))
            except Exception as exc:
                logger.warning("Failed to read %s: %s", path, exc)
    return articles


def build_dataset_from_jsonl(jsonl_path: str) -> List[Dict]:
    """
    Load a pre-labelled JSONL file where each line is:
        {"article": "...", "summary": "..."}
    """
    records: List[Dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if "article" in record and "summary" in record:
                    records.append(record)
                else:
                    logger.warning("Line %d missing 'article'/'summary' keys", line_no)
            except json.JSONDecodeError as exc:
                logger.warning("Invalid JSON on line %d: %s", line_no, exc)
    logger.info("Loaded %d records from %s", len(records), jsonl_path)
    return records


def preprocess_and_save(
    records: List[Dict],
    output_path: str,
    tokenizer_name: str = "facebook/bart-large-cnn",
    max_input_length: int = 1024,
    max_target_length: int = 256,
) -> None:
    """
    Clean, tokenize, and persist processed records as JSONL.

    Args:
        records: List of {"article": ..., "summary": ...} dicts.
        output_path: Destination .jsonl file path.
        tokenizer_name: HuggingFace model / tokenizer identifier.
        max_input_length: Max tokens for the source article.
        max_target_length: Max tokens for the target summary.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    ensure_dir(os.path.dirname(output_path))

    with open(output_path, "w", encoding="utf-8") as out:
        for rec in tqdm(records, desc="Preprocessing"):
            article = clean_text(rec["article"])
            summary = clean_text(rec["summary"])

            if not article or not summary:
                continue

            tokenized = tokenize_pair(
                article, summary, tokenizer,
                max_input_length=max_input_length,
                max_target_length=max_target_length,
            )

            # Also store the raw text for reference
            tokenized["raw_article"] = article
            tokenized["raw_summary"] = summary

            out.write(json.dumps(tokenized) + "\n")

    logger.info("Saved processed data to %s", output_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess articles for summarization")
    parser.add_argument("--input", required=True, help="Path to input JSONL or raw-article directory")
    parser.add_argument("--output", required=True, help="Path to output JSONL file")
    parser.add_argument("--tokenizer", default="facebook/bart-large-cnn", help="HuggingFace tokenizer name")
    parser.add_argument("--max-input-length", type=int, default=1024)
    parser.add_argument("--max-target-length", type=int, default=256)
    args = parser.parse_args()

    if args.input.endswith(".jsonl"):
        records = build_dataset_from_jsonl(args.input)
    else:
        raw = load_raw_articles(args.input)
        # When loading from raw files the user must supply summaries separately;
        # for now we simply store articles with empty summaries as a placeholder.
        records = [{"article": text, "summary": ""} for _, text in raw]
        logger.warning(
            "Raw-file mode: summaries are empty – add them before training."
        )

    preprocess_and_save(
        records,
        args.output,
        tokenizer_name=args.tokenizer,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
    )


if __name__ == "__main__":
    main()
