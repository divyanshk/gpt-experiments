import torch
import tiktoken
from models.gpt2_model import GPTModel, text_to_token_ids, token_ids_to_text, generate
from configs.config import get_config

# Get local configuration for development
model_config, training_config = get_config("local")

# Initialize model
model = GPTModel(model_config)
device = torch.device(training_config["device"])
model.to(device)

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Test text generation
prompt = "Explain quantum mechanics clearly and concisely."
print(f"Prompt: {prompt}")

# Convert text to tokens
token_ids = text_to_token_ids(prompt, tokenizer).to(device)

# Generate text
with torch.no_grad():
    generated_ids = generate(
        model=model,
        idx=token_ids,
        max_new_tokens=50,
        context_size=model_config["context_length"],
        temperature=0.7,
        top_k=20
    )

# Convert back to text
generated_text = token_ids_to_text(generated_ids, tokenizer)
print(f"Generated: {generated_text}")

