"""
Training CLI for the Article Summarization System.

Usage:
    python -m src.train \
        --data_dir data/processed \
        --output_dir models \
        --epochs 3 \
        --batch_size 4 \
        --learning_rate 5e-5
"""

import argparse
import json
import os
from typing import List, Dict

from datasets import Dataset

from src.model import SummarizationModel
from src.utils import setup_logging, set_seed

logger = setup_logging()


def load_jsonl(path: str) -> List[Dict]:
    """Load tokenized JSONL records."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def records_to_dataset(records: List[Dict]) -> Dataset:
    """
    Convert a list of tokenized record dicts into a HuggingFace Dataset.

    Expects each record to have at least ``input_ids``, ``attention_mask``,
    and ``labels`` keys (lists of ints).
    """
    columns = {
        "input_ids": [r["input_ids"] for r in records],
        "attention_mask": [r["attention_mask"] for r in records],
        "labels": [r["labels"] for r in records],
    }
    return Dataset.from_dict(columns)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the summarization model")
    parser.add_argument("--data_dir", required=True, help="Directory with train.jsonl (and optional val.jsonl)")
    parser.add_argument("--output_dir", default="models", help="Where to save checkpoints")
    parser.add_argument("--model_name", default="facebook/bart-large-cnn", help="HuggingFace model identifier")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--fp16", action="store_true", help="Use mixed-precision training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_input_length", type=int, default=1024)
    parser.add_argument("--max_target_length", type=int, default=256)
    args = parser.parse_args()

    set_seed(args.seed)

    # ---- Load data --------------------------------------------------------
    train_path = os.path.join(args.data_dir, "train.jsonl")
    val_path = os.path.join(args.data_dir, "val.jsonl")

    if not os.path.isfile(train_path):
        raise FileNotFoundError(f"Training data not found at {train_path}")

    logger.info("Loading training data from %s", train_path)
    train_records = load_jsonl(train_path)
    train_dataset = records_to_dataset(train_records)
    logger.info("Training set: %d examples", len(train_dataset))

    val_dataset = None
    if os.path.isfile(val_path):
        logger.info("Loading validation data from %s", val_path)
        val_records = load_jsonl(val_path)
        val_dataset = records_to_dataset(val_records)
        logger.info("Validation set: %d examples", len(val_dataset))

    # ---- Build & train model ----------------------------------------------
    model = SummarizationModel(
        model_name=args.model_name,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
        seed=args.seed,
    )

    model.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
    )

    logger.info("Training complete ✓")


if __name__ == "__main__":
    main()
