
import os
import torch
import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_scheduler
from utils import get_device, load_prompts, tokenize_prompts

def main(data_path):
    # Parameters
    model_name = 'gpt2'
    max_length = 128
    batch_size = 4
    epochs = 3
    learning_rate = 5e-5

    # Load Dataset
    prompts = load_prompts(data_path)

    # Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Tokenize Data
    inputs = tokenize_prompts(prompts, tokenizer, max_length)

    # DataLoader
    dataset = torch.utils.data.TensorDataset(inputs['input_ids'])
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    device = get_device()
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.to(device)

    # Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = len(train_loader) * epochs
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # Training Function
    def train_epoch(model, data_loader, optimizer, device, scheduler):
        model.train()
        total_loss = 0

        for batch in data_loader:
            input_ids = batch[0].to(device)

            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(data_loader)
        return avg_loss

    # Training Loop
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, lr_scheduler)
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss}')

    # Save Model
    model_dir = './models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the text file containing prompts')
    args = parser.parse_args()
    main(args.data_path)
