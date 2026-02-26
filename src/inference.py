"""
Inference API for the Article Summarization System.

Exposes a FastAPI server with a ``POST /summarize`` endpoint that accepts
an article and returns a generated summary.

Usage:
    uvicorn src.inference:app --reload --port 8000
"""

import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.model import SummarizationModel
from src.utils import setup_logging

logger = setup_logging()

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Article Summarization API",
    description="Generate coherent summaries of long-form articles using Transformer models.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model loading (lazy singleton)
# ---------------------------------------------------------------------------

_model: Optional[SummarizationModel] = None


def get_model() -> SummarizationModel:
    """Return the loaded model, creating it on first call."""
    global _model
    if _model is None:
        model_dir = os.environ.get("MODEL_DIR", "models/best_model")
        if os.path.isdir(model_dir):
            logger.info("Loading fine-tuned model from %s", model_dir)
            _model = SummarizationModel.from_pretrained(model_dir)
        else:
            # Fall back to the pretrained model for quick testing
            model_name = os.environ.get("MODEL_NAME", "facebook/bart-large-cnn")
            logger.info("No fine-tuned model found – using pretrained %s", model_name)
            _model = SummarizationModel(model_name=model_name)
    return _model


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class SummarizeRequest(BaseModel):
    article: str = Field(
        ...,
        min_length=10,
        description="The full article text to summarize.",
    )
    max_length: Optional[int] = Field(
        None,
        ge=10,
        le=1024,
        description="Maximum number of tokens in the generated summary.",
    )
    min_length: int = Field(
        56,
        ge=1,
        description="Minimum number of tokens in the generated summary.",
    )
    num_beams: int = Field(
        4, ge=1, le=10,
        description="Number of beams for beam search.",
    )


class SummarizeResponse(BaseModel):
    summary: str
    article_length: int
    summary_length: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    """Minimal HTML UI for quick testing."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Article Summarizer</title>
        <style>
            :root {
                --bg: #0f172a;
                --surface: #1e293b;
                --primary: #818cf8;
                --primary-hover: #6366f1;
                --text: #e2e8f0;
                --muted: #94a3b8;
                --border: #334155;
                --radius: 12px;
            }
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body {
                font-family: 'Inter', 'Segoe UI', sans-serif;
                background: var(--bg);
                color: var(--text);
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 2rem 1rem;
            }
            h1 {
                font-size: 2rem;
                background: linear-gradient(135deg, var(--primary), #a78bfa);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: .5rem;
            }
            p.subtitle { color: var(--muted); margin-bottom: 2rem; }
            .container {
                width: 100%;
                max-width: 800px;
                display: flex;
                flex-direction: column;
                gap: 1.5rem;
            }
            textarea {
                width: 100%;
                min-height: 200px;
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: var(--radius);
                color: var(--text);
                padding: 1rem;
                font-size: 1rem;
                resize: vertical;
                transition: border-color .2s;
            }
            textarea:focus {
                outline: none;
                border-color: var(--primary);
            }
            button {
                padding: .85rem 2rem;
                border: none;
                border-radius: var(--radius);
                background: var(--primary);
                color: #fff;
                font-size: 1rem;
                font-weight: 600;
                cursor: pointer;
                transition: background .2s, transform .1s;
            }
            button:hover { background: var(--primary-hover); }
            button:active { transform: scale(.98); }
            button:disabled {
                opacity: .5;
                cursor: not-allowed;
            }
            #result {
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: var(--radius);
                padding: 1.25rem;
                white-space: pre-wrap;
                display: none;
            }
            .meta { color: var(--muted); font-size: .85rem; margin-top: .75rem; }
            .spinner {
                display: inline-block;
                width: 18px; height: 18px;
                border: 2px solid rgba(255,255,255,.3);
                border-top-color: #fff;
                border-radius: 50%;
                animation: spin .6s linear infinite;
                vertical-align: middle;
                margin-right: .5rem;
            }
            @keyframes spin { to { transform: rotate(360deg); } }
        </style>
    </head>
    <body>
        <h1>📝 Article Summarizer</h1>
        <p class="subtitle">Paste an article below and get a concise AI-generated summary.</p>
        <div class="container">
            <textarea id="article" placeholder="Paste your article text here…"></textarea>
            <button id="btn" onclick="summarize()">Summarize</button>
            <div id="result"></div>
        </div>
        <script>
            async function summarize() {
                const btn = document.getElementById('btn');
                const resultDiv = document.getElementById('result');
                const article = document.getElementById('article').value.trim();
                if (!article) return;

                btn.disabled = true;
                btn.innerHTML = '<span class="spinner"></span>Generating…';
                resultDiv.style.display = 'none';

                try {
                    const res = await fetch('/summarize', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ article }),
                    });
                    if (!res.ok) throw new Error(await res.text());
                    const data = await res.json();
                    resultDiv.innerHTML =
                        '<strong>Summary</strong><br><br>' +
                        data.summary +
                        '<div class="meta">Article: ' + data.article_length +
                        ' chars → Summary: ' + data.summary_length + ' chars</div>';
                    resultDiv.style.display = 'block';
                } catch (err) {
                    resultDiv.innerHTML = '<span style="color:#f87171;">Error: ' + err.message + '</span>';
                    resultDiv.style.display = 'block';
                } finally {
                    btn.disabled = false;
                    btn.textContent = 'Summarize';
                }
            }
        </script>
    </body>
    </html>
    """


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    """Generate a summary for the provided article text."""
    try:
        model = get_model()
        summary = model.summarize(
            text=request.article,
            max_length=request.max_length,
            min_length=request.min_length,
            num_beams=request.num_beams,
        )
        return SummarizeResponse(
            summary=summary,
            article_length=len(request.article),
            summary_length=len(summary),
        )
    except Exception as exc:
        logger.exception("Summarization failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
async def health():
    """Simple health-check endpoint."""
    return {"status": "ok"}
