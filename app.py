import streamlit as st
import torch
import tiktoken
import sys
import os
from huggingface_hub import hf_hub_download

sys.path.append(os.path.dirname(__file__))

from src.inference import classify_review, load_model
from src.config import GPT_CONFIG_124M

st.set_page_config(
    page_title="Spam Classifier",
    page_icon="🛡️",
    layout="centered"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }

    /* Top bar */
    .topbar {
        border-bottom: 1px solid #21262d;
        padding-bottom: 1.2rem;
        margin-bottom: 2.5rem;
    }
    .topbar-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        color: #3fb950;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .topbar-title {
        font-size: 1.6rem;
        font-weight: 600;
        color: #e6edf3;
        letter-spacing: -0.02em;
    }
    .topbar-meta {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        color: #8b949e;
        margin-top: 0.4rem;
    }

    /* Model info strip */
    .info-strip {
        display: flex;
        gap: 2rem;
        margin-bottom: 2rem;
        padding: 1rem 0;
        border-top: 1px solid #21262d;
        border-bottom: 1px solid #21262d;
    }
    .info-item {
        display: flex;
        flex-direction: column;
        gap: 0.2rem;
    }
    .info-key {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.65rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .info-val {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        color: #58a6ff;
        font-weight: 600;
    }

    /* Input label */
    .input-label {
        font-size: 0.82rem;
        font-weight: 500;
        color: #8b949e;
        margin-bottom: 0.4rem;
        letter-spacing: 0.01em;
    }

    /* Result */
    .result-box {
        margin-top: 1.5rem;
        padding: 1.2rem 1.5rem;
        border-radius: 6px;
        display: flex;
        align-items: flex-start;
        gap: 1rem;
    }
    .result-spam {
        background-color: rgba(248, 81, 73, 0.08);
        border: 1px solid rgba(248, 81, 73, 0.4);
    }
    .result-ham {
        background-color: rgba(63, 185, 80, 0.08);
        border: 1px solid rgba(63, 185, 80, 0.35);
    }
    .result-icon {
        font-size: 1.3rem;
        margin-top: 0.1rem;
    }
    .result-heading {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    .result-spam .result-heading { color: #f85149; }
    .result-ham  .result-heading { color: #3fb950; }
    .result-desc {
        font-size: 0.82rem;
        color: #8b949e;
        line-height: 1.5;
    }

    /* Footer */
    .footer {
        margin-top: 3rem;
        padding-top: 1.2rem;
        border-top: 1px solid #21262d;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    .footer-left {
        font-size: 0.78rem;
        color: #8b949e;
    }
    .footer-right {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        color: #3fb950;
    }
    .footer a {
        color: #58a6ff;
        text-decoration: none;
    }
    .footer a:hover {
        text-decoration: underline;
    }

    /* Hide streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 2.5rem; max-width: 680px; }
</style>
""", unsafe_allow_html=True)

# ── Load model ──────────────────────────────────────────────
@st.cache_resource
def get_model():
    weights_path = hf_hub_download(
        repo_id="uniquehere00/gpt2-spam-classifier",
        filename="spam_classifier_gpu.pth",
        repo_type="model"
    )
    device = torch.device("cpu")
    model = load_model(
        model_path=weights_path,
        config=GPT_CONFIG_124M,
        num_classes=2,
        device=device
    )
    return model, device

model, device = get_model()
tokenizer = tiktoken.get_encoding("gpt2")

# ── Header ──────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <div class="topbar-label">● model · deployed</div>
    <div class="topbar-title">SMS Spam Classifier</div>
    <div class="topbar-meta">uniquehere00/gpt2-spam-classifier</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-strip">
    <div class="info-item">
        <span class="info-key">Architecture</span>
        <span class="info-val">GPT-2 124M</span>
    </div>
    <div class="info-item">
        <span class="info-key">Test Accuracy</span>
        <span class="info-val">94.76%</span>
    </div>
    <div class="info-item">
        <span class="info-key">Dataset</span>
        <span class="info-val">SMS Spam Collection</span>
    </div>
    <div class="info-item">
        <span class="info-key">Framework</span>
        <span class="info-val">PyTorch</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Input ───────────────────────────────────────────────────
st.markdown('<div class="input-label">Enter SMS message</div>', unsafe_allow_html=True)

text = st.text_area(
    label="",
    height=130,
    placeholder="Paste an SMS message here...",
    label_visibility="collapsed"
)

clicked = st.button("Run classifier →", type="primary", use_container_width=True)

# ── Output ──────────────────────────────────────────────────
if clicked:
    if text.strip() == "":
        st.warning("No input detected. Please paste a message.")
    else:
        with st.spinner("Running inference..."):
            result = classify_review(
                text=text,
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_length=120
            )

        if result == "spam":
            st.markdown("""
            <div class="result-box result-spam">
                <div class="result-icon">⚠</div>
                <div>
                    <div class="result-heading">SPAM DETECTED</div>
                    <div class="result-desc">
                        This message matches patterns commonly associated with spam or phishing.
                        Do not click any links or share personal information.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-box result-ham">
                <div class="result-icon">✓</div>
                <div>
                    <div class="result-heading">NOT SPAM</div>
                    <div class="result-desc">
                        This message appears to be legitimate.
                        No spam signals were detected.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ── Footer ──────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="footer-left">
        Built by <strong>Unique Das</strong> ·
        <a href="https://github.com/uniquehere00/gpt2-spam-classifier">GitHub</a> ·
        <a href="https://huggingface.co/uniquehere00/gpt2-spam-classifier">HuggingFace</a>
    </div>
    <div class="footer-right">GPT-2 from scratch · PyTorch</div>
</div>
""", unsafe_allow_html=True)