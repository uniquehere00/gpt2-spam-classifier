#Model Configuration
GPT_CONFIG_124M = {
    "vocab_size": 50257,   # Vocabulary size
    "context_length": 1024, # Shortened context length (orig: 1024)
    "emb_dim": 768,        # Embedding dimension
    "n_heads": 12,         # Number of attention heads
    "n_layers": 12,        # Number of layers
    "drop_rate": 0.1,      # Dropout rate
    "qkv_bias": True      # Query-key-value bias
}

# Training Configuration
TRAINING_CONFIG = {
    "lr": 2e-5,               # Learning rate
    "weight_decay": 0.1,      # Weight decay for AdamW
    "num_epochs": 5,          # Number of training epochs
    "batch_size": 8,          # Batch size
    "eval_freq": 50,          # Evaluate every N steps
    "eval_iter": 5,           # Number of batches for evaluation
}

# Data Paths
TRAIN_PATH = "data/train.xls"
VAL_PATH = "data/validation.xls"
TEST_PATH = "data/test.xls"

#Model Paths
MODEL_SAVE_PATH = "saved_model/spam_classifier_gpu.pth"

# Classification
NUM_CLASSES = 2