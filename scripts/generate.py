import torch
import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from utils import get_device

def main(model_path, prompt):
    # Load Model and Tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    # Device
    device = get_device()
    model.to(device)

    # Generate Text
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    output = model.generate(input_ids, max_length=128, num_return_sequences=1, no_repeat_ngram_size=2)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(generated_text)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the fine-tuned model')
    parser.add_argument('--prompt', type=str, required=True, help='Prompt for text generation')
    args = parser.parse_args()
    main(args.model_path, args.prompt)
