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
    page_title="SMS Spam Classifier",
    page_icon="🛡️",
    layout="centered"
)

# ── Custom CSS ─────────────────────────────────
st.markdown("""
<style>
    /* Background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    }

    /* Main card */
    .main-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2.5rem;
        backdrop-filter: blur(10px);
        margin-bottom: 1.5rem;
    }

    /* Title */
    .title {
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    /* Stat badges */
    .stats-row {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin-bottom: 2rem;
        flex-wrap: wrap;
    }
    .stat-badge {
        background: rgba(167, 139, 250, 0.15);
        border: 1px solid rgba(167, 139, 250, 0.3);
        border-radius: 50px;
        padding: 0.4rem 1.2rem;
        color: #c4b5fd;
        font-size: 0.85rem;
        font-weight: 600;
    }

    /* Result boxes */
    .result-spam {
        background: linear-gradient(135deg, rgba(239,68,68,0.2), rgba(185,28,28,0.2));
        border: 1px solid rgba(239, 68, 68, 0.5);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        text-align: center;
        margin-top: 1.5rem;
    }
    .result-ham {
        background: linear-gradient(135deg, rgba(34,197,94,0.2), rgba(21,128,61,0.2));
        border: 1px solid rgba(34, 197, 94, 0.5);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        text-align: center;
        margin-top: 1.5rem;
    }
    .result-title {
        font-size: 1.6rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }
    .result-sub {
        font-size: 0.9rem;
        opacity: 0.75;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #475569;
        font-size: 0.8rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255,255,255,0.07);
    }
    .footer a {
        color: #7c3aed;
        text-decoration: none;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────
@st.cache_resource
def get_model():
    with st.spinner("Loading model weights..."):
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

# ── Header ─────────────────────────────────────
st.markdown('<div class="title">🛡️ SMS Spam Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by fine-tuned GPT-2 · Built from scratch in PyTorch</div>', unsafe_allow_html=True)

st.markdown("""
<div class="stats-row">
    <div class="stat-badge">📊 94.76% Accuracy</div>
    <div class="stat-badge">⚡ 124M Parameters</div>
    <div class="stat-badge">🗂️ SMS Spam Collection</div>
</div>
""", unsafe_allow_html=True)

# ── Main card ──────────────────────────────────
st.markdown('<div class="main-card">', unsafe_allow_html=True)

text = st.text_area(
    "Enter SMS message to classify:",
    height=140,
    placeholder="Paste any SMS here — e.g. 'Congratulations! You've won a free iPhone. Click now to claim!'",
    label_visibility="visible"
)

classify_btn = st.button(
    "🔍 Classify Message",
    type="primary",
    use_container_width=True
)

if classify_btn:
    if text.strip() == "":
        st.warning("⚠️ Please enter a message first.")
    else:
        with st.spinner("Analysing..."):
            result = classify_review(
                text=text,
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_length=120
            )

        if result == "spam":
            st.markdown("""
            <div class="result-spam">
                <div class="result-title">🚨 SPAM DETECTED</div>
                <div class="result-sub">This message shows characteristics of spam or phishing.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-ham">
                <div class="result-title">✅ LEGITIMATE MESSAGE</div>
                <div class="result-sub">This message appears to be safe and genuine.</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────
st.markdown("""
<div class="footer">
    Built by <strong>Unique Das</strong> · 
    <a href="https://github.com/uniquehere00/gpt2-spam-classifier">GitHub</a> · 
    <a href="https://huggingface.co/uniquehere00/gpt2-spam-classifier">HuggingFace</a>
</div>
""", unsafe_allow_html=True)