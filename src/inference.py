import torch 
from src.model import GPTModel

def classify_review(text, model, tokenizer, device, max_length, pad_token_id=50256):
    model.eval()

    # Prepare inputs to the model
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]

    # Truncate sequences if they too long
    input_ids = input_ids[:min(max_length, supported_context_length)]

    # Pad sequences to the longest sequence
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0) # add batch dimension

    # Model inference
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]  # Logits of the last output token
    predicted_label = torch.argmax(logits, dim=-1).item()

    # Return the classified result
    return "spam" if predicted_label == 1 else "not spam"

def load_model(model_path, config, num_classes, device):
    model = GPTModel(config)
    model.out_head = torch.nn.Linear(config["emb_dim"], num_classes)
    model.load_state_dict(torch.load(model_path, map_location =device,weights_only =True))
    model.to(device)
    model.eval()
    return model