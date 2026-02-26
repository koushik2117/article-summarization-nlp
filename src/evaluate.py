"""
Evaluation Module for the Article Summarization System.

Loads a trained model checkpoint, runs inference on a held-out dataset,
computes ROUGE metrics, and performs a basic bias/integrity check on
generated summaries.
"""

import argparse
import json
import os
from collections import Counter
from typing import Dict, List, Tuple

from tqdm import tqdm

from src.model import SummarizationModel, compute_rouge
from src.utils import setup_logging

logger = setup_logging()

# ---------------------------------------------------------------------------
# Bias / data-integrity helpers
# ---------------------------------------------------------------------------

# Demographic & sensitive terms to monitor for over-/under-representation
SENSITIVE_TERMS = [
    "he", "she", "man", "woman", "men", "women",
    "black", "white", "asian", "latino", "hispanic",
    "christian", "muslim", "jewish", "hindu",
    "young", "old", "elderly",
]


def _word_freq(texts: List[str]) -> Counter:
    """Build a case-insensitive word frequency counter."""
    counter: Counter = Counter()
    for text in texts:
        words = text.lower().split()
        counter.update(words)
    return counter


def check_bias(
    sources: List[str],
    summaries: List[str],
    threshold: float = 2.0,
) -> List[Dict]:
    """
    Compare word-frequency distributions in source articles vs. generated
    summaries for a set of sensitive / demographic terms.

    A term is *flagged* when its relative frequency in the summaries is more
    than ``threshold`` times higher than in the sources (or vice-versa).

    Returns a list of flag dicts with ``term``, ``source_freq``, ``summary_freq``,
    and ``ratio``.
    """
    src_freq = _word_freq(sources)
    sum_freq = _word_freq(summaries)

    src_total = max(sum(src_freq.values()), 1)
    sum_total = max(sum(sum_freq.values()), 1)

    flags: List[Dict] = []
    for term in SENSITIVE_TERMS:
        src_rate = src_freq.get(term, 0) / src_total
        sum_rate = sum_freq.get(term, 0) / sum_total

        if src_rate == 0 and sum_rate == 0:
            continue

        denom = max(src_rate, 1e-9)
        ratio = sum_rate / denom

        if ratio > threshold or (ratio > 0 and ratio < 1 / threshold):
            flags.append({
                "term": term,
                "source_freq": round(src_rate, 6),
                "summary_freq": round(sum_rate, 6),
                "ratio": round(ratio, 4),
            })

    return flags


def check_data_integrity(records: List[Dict]) -> Dict[str, int]:
    """
    Run basic integrity checks on the evaluation dataset.

    Returns counts of issues found:
    - ``empty_article``: records with blank articles.
    - ``empty_summary``: records with blank summaries.
    - ``duplicates``: duplicate article texts.
    """
    issues = {"empty_article": 0, "empty_summary": 0, "duplicates": 0}
    seen_articles = set()

    for rec in records:
        art = rec.get("article", "").strip()
        summ = rec.get("summary", "").strip()

        if not art:
            issues["empty_article"] += 1
        if not summ:
            issues["empty_summary"] += 1
        if art in seen_articles:
            issues["duplicates"] += 1
        seen_articles.add(art)

    return issues


# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------

def evaluate(
    model: SummarizationModel,
    records: List[Dict],
) -> Tuple[Dict[str, float], List[Dict]]:
    """
    Evaluate a model on ``records`` (list of {"article", "summary"} dicts).

    Returns:
        rouge_scores: Dictionary of ROUGE-1/2/L F1 scores.
        bias_flags: List of flagged bias terms.
    """
    sources = [r["article"] for r in records]
    references = [r["summary"] for r in records]

    logger.info("Generating summaries for %d examples…", len(records))
    predictions: List[str] = []
    for article in tqdm(sources, desc="Summarizing"):
        predictions.append(model.summarize(article))

    # ROUGE metrics
    rouge_scores = compute_rouge(predictions, references)
    logger.info("ROUGE scores:")
    for metric, score in rouge_scores.items():
        logger.info("  %-10s %.4f", metric, score)

    # Bias check
    bias_flags = check_bias(sources, predictions)
    if bias_flags:
        logger.warning("⚠  Bias flags detected:")
        for flag in bias_flags:
            logger.warning("  term=%-12s src_freq=%.6f  sum_freq=%.6f  ratio=%.4f",
                           flag["term"], flag["source_freq"],
                           flag["summary_freq"], flag["ratio"])
    else:
        logger.info("No bias flags detected ✓")

    return rouge_scores, bias_flags


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained summarization model")
    parser.add_argument("--model_dir", required=True, help="Path to saved model directory")
    parser.add_argument("--data_path", required=True, help="Path to evaluation JSONL file")
    parser.add_argument("--output", default=None, help="Optional path to save results JSON")
    args = parser.parse_args()

    # Load model
    model = SummarizationModel.from_pretrained(args.model_dir)

    # Load data
    records: List[Dict] = []
    with open(args.data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "raw_article" in rec:
                rec["article"] = rec["raw_article"]
                rec["summary"] = rec["raw_summary"]
            records.append(rec)

    logger.info("Loaded %d evaluation records from %s", len(records), args.data_path)

    # Data integrity
    integrity = check_data_integrity(records)
    if any(v > 0 for v in integrity.values()):
        logger.warning("Data integrity issues: %s", integrity)
    else:
        logger.info("Data integrity check passed ✓")

    # Evaluate
    rouge_scores, bias_flags = evaluate(model, records)

    # Optionally save
    if args.output:
        results = {
            "rouge": rouge_scores,
            "bias_flags": bias_flags,
            "data_integrity": integrity,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
