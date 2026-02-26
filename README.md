# Article Summarization System (NLP)

[![Live Demo on Hugging Face Spaces](https://img.shields.io/badge/🤗%20Live%20Demo-Hugging%20Face%20Spaces-blue)](https://huggingface.co/spaces/koushik2117/article-summarization-nlp)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-BART--large--CNN-orange)](https://huggingface.co/facebook/bart-large-cnn)

An end-to-end NLP pipeline for generating coherent summaries of long-form articles using Transformer-based models (BART / T5).

> **🔗 [Try the Live Demo →](https://huggingface.co/spaces/koushik2117/article-summarization-nlp)**

## Project Structure

```
article-summarizer/
├─ data/
│   └─ sample_data.jsonl     # 10-example benchmark dataset
├─ src/
│   ├─ utils.py              # Seed, device, logging helpers
│   ├─ data_preprocess.py    # Multi-format readers, cleaning, tokenization
│   ├─ model.py              # SummarizationModel (train + inference)
│   ├─ train.py              # CLI for fine-tuning
│   ├─ evaluate.py           # ROUGE scoring + bias / integrity checks
│   └─ inference.py          # FastAPI server with built-in UI
├─ tests/                    # Pytest test suite
├─ requirements.txt
├─ .gitignore
└─ README.md
```

## Quick Start

### 1. Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Preprocess data

Prepare your data as a JSONL file where each line contains `{"article": "...", "summary": "..."}`. A sample dataset is provided at `data/sample_data.jsonl`.

```bash
python -m src.data_preprocess \
    --input data/sample_data.jsonl \
    --output data/processed/train.jsonl
```

### 4. Train the model

```bash
python -m src.train \
    --data_dir data/processed \
    --output_dir models \
    --epochs 3 \
    --batch_size 4 \
    --learning_rate 5e-5
```

> **Note:** Training on the full BART-large-CNN model requires a GPU with ≥12 GB VRAM. For CPU-only machines, use `--model_name sshleifer/distilbart-cnn-12-6` and reduce batch size.

### 5. Evaluate

```bash
python -m src.evaluate \
    --model_dir models/best_model \
    --data_path data/processed/val.jsonl \
    --output results.json
```

This prints ROUGE-1/2/L scores and any bias flags to the console.

### 6. Run the API server

```bash
uvicorn src.inference:app --reload --port 8000
```

- **UI**: Open [http://localhost:8000](http://localhost:8000) for the built-in web interface.
- **API**: `POST http://localhost:8000/summarize` with body `{"article": "..."}`.
- **Health**: `GET http://localhost:8000/health`.

> By default, if no fine-tuned model is found in `models/best_model`, the API loads the pretrained `facebook/bart-large-cnn` directly.

### 7. Run tests

```bash
pytest tests/ -v
```

## Key Features

| Feature | Description |
|---|---|
| **Multi-format ingestion** | Reads `.txt`, `.html`, `.pdf` articles |
| **Transformer fine-tuning** | Wraps HuggingFace `Seq2SeqTrainer` with ROUGE metrics |
| **Hyperparameter tuning** | Configurable epochs, LR, warmup, weight decay, FP16 |
| **Bias detection** | Flags demographic term imbalance between source and summary |
| **Data integrity checks** | Detects empty fields, duplicates, malformed records |
| **FastAPI server** | Production-ready API with Pydantic validation and CORS |
| **Built-in UI** | Dark-themed web interface for quick testing |

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_DIR` | `models/best_model` | Path to a fine-tuned model directory |
| `MODEL_NAME` | `facebook/bart-large-cnn` | Fallback HuggingFace model name |

## Metrics

The system evaluates summaries using:

- **ROUGE-1** – Unigram overlap
- **ROUGE-2** – Bigram overlap
- **ROUGE-L** – Longest common subsequence

## License

This project was developed as part of academic work at Bharath University (Nov–Dec 2023).
