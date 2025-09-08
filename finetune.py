import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
import pandas as pd
import os
from accelerate import Accelerator
import numpy as np
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F
from transformers import get_scheduler
from dataset_utils import get_dataset, extract_label_from_output, VALID_CLASSES
from evaluation import test_model
from datasets import Dataset
from torch.utils.data import DataLoader

hf_model_id = "meta-llama/Llama-3.2-1B-Instruct"
saved_model_id = "peft_1"
LOGGING_STEPS = 20

num_training_datapoints = 600
num_testing_datapoints = 30
batch_size = 8
epochs = 50  # Number of training epochs
learning_rate = 1e-4  # Learning rate for fine-tuning
gradient_accumulation_steps = 2  # Number of steps to accumulate gradients

clip_value = 1.0


output_dir = os.path.join(f"models", saved_model_id)
best_model_dir = os.path.join(output_dir, "best")
latest_model_dir = os.path.join(output_dir, "latest")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(best_model_dir, exist_ok=True)
os.makedirs(latest_model_dir, exist_ok=True)

# Initialize Accelerator
accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
#device = "mps" # change to your device (cuda or cpu)
device = "cuda" # change to your device (cuda or cpu)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(hf_model_id, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(hf_model_id)
model = model.to(device)
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=['q_proj', 'v_proj']# r".*layers\.(11|12)\..*\.([vq]_proj)"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

train_df, test_df = get_dataset(tokenizer=tokenizer, train_size=num_training_datapoints, test_size=num_testing_datapoints)
print(f"Training Samples: {len(train_df)}, Test Samples: {len(test_df)}")

def tokenize_function(example):
    # Concatenate input and output for training
    input_text = example['prompt']
    output_text = [" " + o + tokenizer.eos_token for o in example['labels']]

    full_text_sequence = [i + o for (i, o ) in zip(input_text, output_text)]
    tokenized = tokenizer(full_text_sequence, padding=True, return_tensors="pt", add_special_tokens=False).to(device)["input_ids"]
    out_tokenized = tokenizer(output_text, padding="max_length", max_length=tokenized.shape[1], return_tensors="pt", add_special_tokens=False).to(device)["input_ids"]
    out_tokenized = torch.where(out_tokenized != tokenizer.pad_token_id, out_tokenized, -100)
    out_tokenized[:, -1] = tokenizer.eos_token_id
    tokenized = tokenized[:, :-1].contiguous()
    out_tokenized = out_tokenized[:, 1:].contiguous()
    attention_mask = tokenized != tokenizer.pad_token_id
    return {
        "input_ids": tokenized,
        "attention_mask": attention_mask,
        "labels": out_tokenized
    }

# Prepare training and test messages
input_ids = train_df["prompt"]
train_labels = train_df["labels"]


dataset = Dataset.from_pandas(train_df)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


test_dataset = Dataset.from_pandas(test_df)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


num_training_steps = epochs * len(dataloader)

optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
model, optimizer = accelerator.prepare(model, optimizer)
scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

loss_fn = nn.CrossEntropyLoss(reduction='mean')

def save_model(dir):
    model.save_pretrained(dir)
    tokenizer.save_pretrained(dir)
    print(f"Model saved to {dir}")

    
def calculate_loss(logits, labels,  inputs):
    ce_loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
    return ce_loss

def combine_input_output(input_ids, labels, logits):
    input_ids_rolled = torch.roll(input_ids, shifts=-1, dims=1)
    logits = logits.argmax(axis=-1)
    combined_labels = torch.where(labels == -100, input_ids_rolled, logits)
    output_str = tokenizer.batch_decode(combined_labels)
    return output_str

num_steps = 0
total_loss = []

base_metrics = test_model(test_dataloader, model, tokenizer)
print("\033[31mBefore Training: \n", "\n".join([f"{k}: {v}" for k, v in base_metrics.items()]), "\033[0m")

best_accuracy = 0
for epoch in range(epochs):

    print(f"\n\nEpoch {epoch + 1}/{epochs}")
    
    for batch in dataloader:
        inputs = tokenize_function(batch) 
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        logits = outputs.logits
        
        loss = calculate_loss(logits, inputs["labels"], inputs["input_ids"])
        loss = loss / gradient_accumulation_steps
        accelerator.backward(loss)
        
        if (num_steps + 1) % gradient_accumulation_steps == 0:
            accelerator.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()
        num_steps += 1
        total_loss.append(loss.detach().cpu().item())      
        if num_steps % LOGGING_STEPS == 0:
            combined_labels = combine_input_output(inputs["input_ids"], inputs["labels"], outputs.logits)
            print(f"Train Loss = {np.mean(total_loss):.4f}\n")
            total_loss = []
            save_model(latest_model_dir)
        
    metrics = test_model(test_dataloader, model, tokenizer)
    print(f"\n\nNum Steps: {num_steps}\nTrain Metrics:\n", "\n".join([f"\033[32m{k}: {v:.3f}\033[0m \033[31m(Base {k}: {base_metrics[k]:.3f})\033[0m" for k, v in metrics.items()]))
    if metrics["accuracy"] > best_accuracy:
        save_model(best_model_dir)
        best_accuracy = metrics["accuracy"]
    print()
