import torch
import tiktoken
import time
import os

from src.config import (
    GPT_CONFIG_124M,
    TRAINING_CONFIG,
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    MODEL_SAVE_PATH,
    NUM_CLASSES
)
from src.model import  create_classifier
from src.dataset import SpamDataset, create_dataloaders
from src.train import train_classifier_simple, calc_accuracy_loader
from src.load_weights import download_and_load_gpt2, load_weights_into_gpt
from src.utils import get_device

# Device Setup
device = get_device()

# Tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Load Datasets
train_dataset = SpamDataset(TRAIN_PATH, tokenizer, max_length=None)
val_dataset = SpamDataset(VAL_PATH, tokenizer, max_length=train_dataset.max_length)
test_dataset = SpamDataset(TEST_PATH, tokenizer, max_length=train_dataset.max_length)

# Create Dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    train_dataset, val_dataset, test_dataset,
    batch_size=TRAINING_CONFIG["batch_size"]
)

# Build Model with Classification Head
torch.manual_seed(123)
model = create_classifier(GPT_CONFIG_124M, NUM_CLASSES)

# Download and Load OPENAI Weights
settings, params = download_and_load_gpt2("124M", "gpt2")
load_weights_into_gpt(model, params)
model.to(device)

# Optimizer
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=TRAINING_CONFIG["lr"],
    weight_decay=TRAINING_CONFIG["weight_decay"]
)

# Training
print("Starting training...")
start_time = time.time()
train_classifier_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=TRAINING_CONFIG["num_epochs"],
    eval_freq=TRAINING_CONFIG["eval_freq"],
    eval_iter=TRAINING_CONFIG["eval_iter"]
)
end_time = time.time()
print(f"Training completed in {(end_time - start_time)/60:.2f} minutes.")

# Evaluate
test_accuracy = calc_accuracy_loader(test_loader, model, device)
print(f"Test accuracy: {test_accuracy*100:.2f}%")

# Save Model
os.makedirs("saved_model", exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")