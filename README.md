# GPT-2 Spam Classifier

Fine-tuning a 124M parameter GPT-2 transformer for binary spam classification. Implements selective layer fine-tuning, loading OpenAI pretrained weights, and a modular training pipeline achieving **94.67% test accuracy** on a T4 GPU in under 2 minutes.

---

## Results

| Setting | Accuracy | Time |
|---|---|---|
| CPU baseline | 83.76% | ~40 min |
| Kaggle T4 GPU | **94.67%** | 1.09 min |

Full metrics available in [`results/metrics.json`](results/metrics.json).

---

## Architecture

GPT-2 (124M) is adapted for classification by replacing the language modeling head with a 2-class linear layer. The base transformer architecture is implemented from scratch in PyTorch — no HuggingFace.

```
Input Tokens
    └── Token + Positional Embeddings
        └── 12x TransformerBlock
              ├── Multi-Head Causal Self-Attention
              ├── Layer Norm
              └── Feed Forward Network
        └── Final Layer Norm
        └── Classification Head (768 → 2)
```

**Fine-tuning strategy:** All 12 transformer blocks are frozen. Only the last transformer block, final layer norm, and classification head are trained. This preserves pretrained language representations while adapting the model to the classification task.

---

## Design Decisions

*Why selective fine-tuning?*

We only have ~1500 training samples. Training all 124M parameters on such a 
small dataset would cause the model to memorize training data rather than 
generalize. Freezing the first 11 layers keeps GPT-2's language understanding 
intact while the last layer adapts to spam detection.

*Why does freezing layers prevent overfitting?*

Fewer trainable parameters means less chance of memorizing noise in small 
datasets. We reduced trainable parameters from 124M to ~7M by freezing most 
layers — enough capacity to learn spam patterns, not enough to overfit.
---

## Project Structure

```
├── src/
│   ├── model.py           # GPT-2 architecture from scratch
│   ├── dataset.py         # Data pipeline: download, balance, split, tokenize
│   ├── train.py           # Training loop and loss functions
│   ├── inference.py       # Model loading and single-text classification
│   ├── load_weights.py    # OpenAI pretrained weight loading (TF → PyTorch)
│   ├── utils.py           # Device setup and token utilities
│   ├── config.py          # Model and training hyperparameters
│   └── main.py            # Entry point: end-to-end training pipeline
│
├── data/
│   ├── train.xls          # 1045 samples
│   ├── validation.xls     # 149 samples
│   └── test.xls           # 300 samples
│
├── notebooks/
│   └── spam_classification_gpt2.ipynb   # Experimentation and analysis
│
├── results/
│   └── metrics.json       # Training results and evaluation metrics
│
└── saved_model/           # Place downloaded model weights here
```

---

## Setup

**Requirements**
```
Python 3.9+
CUDA-compatible GPU (recommended)
```

**Install dependencies**
```bash
pip install -r requirements.txt
```

**Train from scratch**
```bash
python src/main.py
```

This will:
1. Download OpenAI GPT-2 (124M) pretrained weights
2. Load the preprocessed dataset from `data/`
3. Fine-tune with selective layer training
4. Evaluate on test set
5. Save model to `saved_model/`

---

## Inference

```python
import torch
import tiktoken
from src.inference import load_model, classify_review
from src.config import GPT_CONFIG_124M, NUM_CLASSES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = tiktoken.get_encoding("gpt2")

model = load_model("saved_model/spam_classifier_gpu.pth", GPT_CONFIG_124M, NUM_CLASSES, device)

result = classify_review("Congratulations! You won a free iPhone. Click here.", model, tokenizer, device, max_length=128)
print(result)  # "spam"
```

**Download pretrained weights:** [HuggingFace](https://huggingface.co/uniquehere00/gpt2-spam-classifier/resolve/main/spam_classifier_gpu.pth) — place in `saved_model/`

---

## Dataset

SMS Spam Collection from the UCI ML Repository. The dataset is class-imbalanced (4825 ham, 747 spam). Ham samples are undersampled to match spam count before splitting.

| Split | Samples |
|---|---|
| Train | 1045 |
| Validation | 149 |
| Test | 300 |

---

## Key Implementation Details

- **No HuggingFace** — GPT-2 architecture built entirely from scratch in PyTorch including multi-head causal self-attention, layer normalization, GELU activation,positional embeddings and token embeddings. 
- **Causal masking** — upper-triangular mask implemented inside `MultiHeadAttention`so that the tokens could not pay attention to the future tokens,which will lead to cheating .
- **Selective fine-tuning** — only ~7M of 124M parameters trained, achieving 10.9% accuracy improvement over CPU baseline
- **Single entry point** — `main.py` runs the full pipeline from weight download to model evaluation in one command
- **Clean inference API** — `inference.py` exposes `load_model` and `classify_review` decoupled from training code, ready for deployment
- **Reproducibility** — random seeds fixed at 123 across PyTorch and dataset shuffling

---

## Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Weight decay | 0.1 |
| Epochs | 5 |
| Batch size | 8 |
| Hardware | Kaggle T4 GPU |

---

## Limitations and Future Work

**Current limitations:**

- **Dataset size** — ~1500 training samples is small for a 124M parameter model. Performance on out-of-distribution spam (phishing, multilingual, adversarial) is untested
- **Model overcapacity** — GPT-2 (124M) is significantly overparameterized for binary SMS classification. A distilled or smaller model would likely achieve comparable accuracy with lower inference cost
- **No hyperparameter sweep** — learning rate, number of unfrozen layers, and batch size were selected based on standard practice rather than systematic search
- **Single dataset** — evaluation is limited to the SMS Spam Collection. Generalization to email spam, social media, or other domains has not been validated

**Potential improvements:**

- Apply **LoRA (Low-Rank Adaptation)** to fine-tune with even fewer trainable parameters while maintaining accuracy
- Run a **full fine-tuning comparison** to quantify the regularization benefit of selective layer training
- Evaluate on a **larger, more diverse dataset** such as SpamAssassin or the Enron email corpus
- Implement **learning rate scheduling** (cosine decay with warmup) for more stable convergence
- Add **confusion matrix and per-class F1 score** to `results/` for a more complete evaluation picture

---

## References

- Vaswani et al. (2017) — [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Raschka (2024) — Build a Large Language Model From Scratch
- SMS Spam Collection — [UCI ML Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
