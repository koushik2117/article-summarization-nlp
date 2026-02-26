"""
Hugging Face Spaces – Gradio App for Article Summarization.

This file powers the live demo on Hugging Face Spaces.
"""

import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# ── Load model directly (avoids pipeline KeyError issues) ────────────────
model_name = "facebook/bart-large-cnn"

print(f"Loading tokenizer: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"Loading model: {model_name}")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def summarize(
    article: str,
    max_length: int = 150,
    min_length: int = 40,
    num_beams: int = 4,
) -> str:
    """Generate a summary using BART-large-CNN (Direct Loading)."""
    if not article or len(article.strip()) < 20:
        return "⚠️ Please paste a longer article (at least a few sentences)."

    try:
        inputs = tokenizer(
            article, 
            max_length=1024, 
            truncation=True, 
            return_tensors="pt"
        ).to(device)
        
        summary_ids = model.generate(
            **inputs,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            no_repeat_ngram_size=3,
            length_penalty=2.0,
        )
        
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        return f"❌ Error during summarization: {str(e)}"


# ── Example articles ─────────────────────────────────────────────────────
examples = [
    [
        "Artificial intelligence (AI) has transformed numerous industries, from healthcare "
        "to finance, by enabling machines to learn from data and make decisions with minimal "
        "human intervention. In healthcare, AI-powered diagnostic tools can analyse medical "
        "images with remarkable accuracy, often matching or surpassing human radiologists. "
        "Machine learning algorithms sift through vast datasets of patient records to identify "
        "patterns that predict disease onset, enabling earlier intervention and improved outcomes.",
        150, 40, 4,
    ],
    [
        "Climate change is one of the most pressing challenges facing humanity today. Rising "
        "global temperatures, driven primarily by the burning of fossil fuels, are causing sea "
        "levels to rise, glaciers to melt, and extreme weather events to become more frequent "
        "and severe. The Intergovernmental Panel on Climate Change (IPCC) has warned that "
        "limiting global warming to 1.5 degrees Celsius above pre-industrial levels is critical "
        "to avoiding the most catastrophic impacts.",
        130, 30, 4,
    ],
]

# ── Gradio UI ─────────────────────────────────────────────────────────────
with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="purple",
        neutral_hue="slate",
    ),
    title="Article Summarizer — NLP",
    css="""
        .gradio-container { max-width: 900px !important; }
        #header { text-align: center; margin-bottom: 0.5rem; }
        #header h1 { 
            background: linear-gradient(135deg, #818cf8, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.2rem;
        }
        #summary-box textarea { font-size: 1rem; line-height: 1.6; }
    """,
) as demo:
    gr.HTML(
        """<div id="header">
            <h1>📝 Article Summarizer</h1>
            <p style="color:#94a3b8;">
                Paste a long-form article and get a coherent, concise summary
                powered by <strong>BART-large-CNN</strong>.
            </p>
        </div>"""
    )

    with gr.Row():
        with gr.Column(scale=3):
            article_input = gr.Textbox(
                label="Article",
                placeholder="Paste your article here…",
                lines=12,
            )
        with gr.Column(scale=1):
            max_len = gr.Slider(50, 300, value=150, step=10, label="Max Length")
            min_len = gr.Slider(10, 100, value=40, step=5, label="Min Length")
            beams = gr.Slider(1, 8, value=4, step=1, label="Beam Width")

    btn = gr.Button("✨ Summarize", variant="primary", size="lg")

    summary_output = gr.Textbox(
        label="Summary",
        lines=6,
        elem_id="summary-box",
        interactive=False,
    )

    btn.click(
        fn=summarize,
        inputs=[article_input, max_len, min_len, beams],
        outputs=summary_output,
    )

    gr.Examples(
        examples=examples,
        inputs=[article_input, max_len, min_len, beams],
        outputs=summary_output,
        fn=summarize,
        cache_examples=False,
    )

    gr.Markdown(
        "---\n"
        "**Model:** `facebook/bart-large-cnn` · "
        "**Built with:** 🤗 Transformers + Gradio\n\n"
        "*Developed as part of academic work at Bharath University (Nov–Dec 2023)*"
    )

if __name__ == "__main__":
    demo.launch()
