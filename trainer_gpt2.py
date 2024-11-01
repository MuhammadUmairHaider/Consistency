from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import torch


dataset_name = "fancyzhx/dbpedia_14"

text_tag = "content"


# Load dataset and tokenizer
dataset = load_dataset(dataset_name)

print(dataset)

# Add special tokens for our task

from datasets import load_dataset
from transformers import GPT2Tokenizer, DataCollatorForLanguageModeling
import random
import numpy as np
import torch  # if you're using PyTorch
# import tensorflow as tf  # if you're using TensorFlow

# Set random seed
seed_value = 42  # or any other integer

random.seed(seed_value)
np.random.seed(seed_value)

if torch.cuda.is_available():  # PyTorch-specific
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

import torch

torch.autograd.set_detect_anomaly(True)
# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

special_tokens_dict = {}
new_tokens = []
label2text = dataset['train'].features['label'].names

for label in label2text:
    # Create special token format (with and without space)
    special_token = f'{label}'
    special_token_with_space = f'[{label}]'
    
    # Check if the label is already a single token in the tokenizer
    label_tokens = tokenizer.encode(label, add_special_tokens=False)
    is_single_token = len(label_tokens) == 1
    
    if is_single_token:
        print(f"'{label}' is already a single token (ID: {label_tokens[0]})")
    
    # Add both versions to new tokens list
    new_tokens.extend([special_token])

# Add the tokens to the tokenizer
num_added_tokens = tokenizer.add_tokens(new_tokens)
print(f"\nAdded {num_added_tokens} new tokens to the tokenizer")

special_tokens = {
    'pad_token': '<|pad|>',
    'sep_token': '<|sep|>',
    'eos_token': '<|eos|>'
}
tokenizer.add_special_tokens(special_tokens)

def format_data(examples):
    formatted_texts = []
    for text, label in zip(examples[text_tag], examples['label']):
        # Convert label to string
        
        tok_text = tokenizer.encode(text, max_length=120, truncation=True)
        text = tokenizer.decode(tok_text)
        label_str = dataset['train'].features['label'].int2str(label)
        formatted_text = f"Classify : {text}{tokenizer.sep_token}{label_str}{tokenizer.eos_token}"
        formatted_texts.append(formatted_text)
    return {'formatted_text': formatted_texts}

def tokenize_and_prepare(examples):

    # Tokenize with batch processing
    tokenized = tokenizer(
        examples['formatted_text'],
        padding='max_length',
        max_length=128,
        truncation=True,
        return_tensors="pt"  # Ensure outputs are returned as tensors
    )
    
    # Clone input_ids to create labels
    labels = tokenized['input_ids'].clone()
    
    # Find the position of sep_token
    sep_token_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
    sep_positions = (labels == sep_token_id).nonzero(as_tuple=True)
    
    # Mask all tokens with -100 except for the token right after sep_token
    labels[:] = -100  # Mask all initially
    for batch_idx, sep_pos in zip(*sep_positions):
        if sep_pos + 1 < labels.size(1):
            labels[batch_idx, sep_pos + 1] = tokenized['input_ids'][batch_idx, sep_pos + 1]
    
    # Set padding tokens to -100
    labels[labels == tokenizer.pad_token_id] = -100
    
    return {
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
        'labels': labels
    }
# Process the dataset
formatted_dataset = dataset.map(format_data, batched=True)
tokenized_dataset = formatted_dataset.map(
    tokenize_and_prepare, 
    batched=True,
    # remove_columns=formatted_dataset['train'].column_names
)

# Initialize model with expanded vocabulary
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))

# Training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-emotion-classifier",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    logging_dir='./logs',
)

# Custom data collator
class GPT2DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, examples):
        batch = {}
        batch["input_ids"] = torch.stack([torch.tensor(example["input_ids"]) for example in examples])
        batch["attention_mask"] = torch.stack([torch.tensor(example["attention_mask"]) for example in examples])
        batch["labels"] = torch.stack([torch.tensor(example["labels"]) for example in examples])
        return batch

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    data_collator=GPT2DataCollator(tokenizer),
    tokenizer=tokenizer,
)

trainer.train()


import os

base_path = os.path.join("model_weights", dataset_name)
if not os.path.exists(base_path):
    os.makedirs(base_path)

weights_path = os.path.join(base_path, "weights.pth")

torch.save(model.state_dict(), weights_path)
# model.load_state_dict(torch.load(weights_path))