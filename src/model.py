"""
Model Definition for the Article Summarization System.

Wraps a HuggingFace Seq2Seq Transformer (default: BART-large-CNN) with
custom metric computation (ROUGE-1/2/L) and convenience helpers for
loading / saving checkpoints.
"""

import os
from typing import Dict, Optional

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from rouge_score import rouge_scorer

from src.utils import setup_logging, get_device, set_seed

logger = setup_logging()


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_rouge(
    predictions: list[str],
    references: list[str],
) -> Dict[str, float]:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

    Args:
        predictions: Generated summary strings.
        references:  Ground-truth summary strings.

    Returns:
        Dictionary with rouge1, rouge2, rougeL F1 scores.
    """
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True,
    )
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        for key in scores:
            scores[key].append(result[key].fmeasure)

    return {k: float(np.mean(v)) for k, v in scores.items()}


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class SummarizationModel:
    """
    High-level wrapper around a HuggingFace seq2seq model for
    abstractive text summarization.
    """

    def __init__(
        self,
        model_name: str = "facebook/bart-large-cnn",
        max_input_length: int = 1024,
        max_target_length: int = 256,
        seed: int = 42,
    ):
        set_seed(seed)
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.device = get_device()

        logger.info("Loading tokenizer: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        logger.info("Loading model: %s → %s", model_name, self.device)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        output_dir: str = "models",
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 5e-5,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        fp16: bool = False,
        logging_steps: int = 50,
        save_strategy: str = "epoch",
    ) -> None:
        """
        Fine-tune the model on *train_dataset* and optionally evaluate on
        *val_dataset* after each epoch.
        """
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
        )

        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            fp16=fp16 and torch.cuda.is_available(),
            logging_steps=logging_steps,
            save_strategy=save_strategy,
            eval_strategy="epoch" if val_dataset else "no",
            predict_with_generate=True,
            generation_max_length=self.max_target_length,
            load_best_model_at_end=bool(val_dataset),
            report_to="none",
        )

        def _compute_metrics(eval_preds):
            preds, labels = eval_preds
            # Replace -100 with pad token id
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            return compute_rouge(decoded_preds, decoded_labels)

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=_compute_metrics,
        )

        logger.info("Starting training for %d epoch(s)…", epochs)
        trainer.train()
        trainer.save_model(os.path.join(output_dir, "best_model"))
        self.tokenizer.save_pretrained(os.path.join(output_dir, "best_model"))
        logger.info("Model saved to %s/best_model", output_dir)

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def summarize(
        self,
        text: str,
        max_length: Optional[int] = None,
        min_length: int = 56,
        num_beams: int = 4,
        length_penalty: float = 2.0,
        no_repeat_ngram_size: int = 3,
    ) -> str:
        """
        Generate a summary for a single article.

        Args:
            text: The source article text.
            max_length: Maximum number of tokens in the output.
            min_length: Minimum number of tokens in the output.
            num_beams: Beam search width.
            length_penalty: Exponential penalty on length.
            no_repeat_ngram_size: Block repeated n-grams of this size.

        Returns:
            The generated summary string.
        """
        max_length = max_length or self.max_target_length

        inputs = self.tokenizer(
            text,
            max_length=self.max_input_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        summary_ids = self.model.generate(
            **inputs,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )

        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    @classmethod
    def from_pretrained(cls, model_dir: str, **kwargs) -> "SummarizationModel":
        """Load a previously fine-tuned model from *model_dir*."""
        instance = cls.__new__(cls)
        instance.device = get_device()
        instance.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        instance.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        instance.model.to(instance.device)
        instance.max_input_length = kwargs.get("max_input_length", 1024)
        instance.max_target_length = kwargs.get("max_target_length", 256)
        instance.model_name = model_dir
        logger.info("Loaded model from %s → %s", model_dir, instance.device)
        return instance
