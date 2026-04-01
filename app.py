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

@st.cache_resource
def get_model():
    weights_path = hf_hub_download(
        repo_id="uniquehere00/gpt2-spam-classifier",
        filename="spam_classifier_gpu.pth"
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

st.title("🛡️ SMS Spam Classifier")
st.write("Fine-tuned GPT-2 (124M parameters) — **94.76% accuracy**")
st.divider()

text = st.text_area(
    "Paste your SMS message here:",
    height=120,
    placeholder="e.g. Congratulations! You've won a free iPhone..."
)

if st.button("Classify", type="primary", use_container_width=True):
    if text.strip() == "":
        st.warning("Please enter a message first.")
    else:
        with st.spinner("Classifying..."):
            result = classify_review(
                text=text,
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_length=120
            )

        if result == "spam":
            st.error("🚨 SPAM detected")
        else:
            st.success("✅ NOT SPAM — looks legitimate")

st.divider()
st.caption(
    "Built by Unique Das · "
    "GPT-2 architecture implemented from scratch in PyTorch · "
    "Trained on SMS Spam Collection dataset"
)