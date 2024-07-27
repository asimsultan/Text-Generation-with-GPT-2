import torch
from transformers import GPT2Tokenizer

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_prompts(data_path):
    with open(data_path, 'r') as file:
        prompts = file.readlines()
    return [prompt.strip() for prompt in prompts]

def tokenize_prompts(prompts, tokenizer, max_length):
    return tokenizer(prompts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
