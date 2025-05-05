"""
Emotion Classification with GPT-2 and Activation Masking

This script implements emotion classification using a GPT-2 model with activation masking
techniques to analyze which features are important for each emotion class.
"""

import os
import random
import numpy as np
import torch
from tqdm import tqdm
from prettytable import PrettyTable
import pandas as pd
import warnings

from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset, Features, ClassLabel, Value
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer, DataCollatorForLanguageModeling, GPT2LMHeadModel as gt

# Import custom modules
from utilities import (
    evaluate_gpt2_classification, 
    mask_range_gpt, 
    compute_masks, 
    reset_gpt, 
    compute_mask_probe, 
    mask_gpt2
)
from models.gpt2 import GPT2LMHeadModel

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.tensor")

# Enable anomaly detection for PyTorch (helps with debugging)
torch.autograd.set_detect_anomaly(True)

# Set random seed for reproducibility
def set_seed(seed_value=42):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    if torch.cuda.is_available():
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

# Dataset processing functions
def sample_balanced_dataset(dataset_dict, max_train_per_class=800, max_test_per_class=200):
    """
    Sample a balanced subset while preserving the original feature structure including ClassLabel.
    """
    # Store original features
    original_features = dataset_dict['train'].features
    
    # Convert to pandas for sampling
    train_df = dataset_dict['train'].to_pandas()
    test_df = dataset_dict['test'].to_pandas()
    
    # Group by label
    train_groups = train_df.groupby('label')
    test_groups = test_df.groupby('label')
    
    sampled_train_dfs = []
    sampled_test_dfs = []
    
    print("\nClass distribution:")
    print("\nLabel | Label Name | Train Samples | Test Samples | Final Train | Final Test")
    print("-" * 85)
    
    label_names = original_features['label'].names
    for idx, label_name in enumerate(label_names):
        train_group = train_groups.get_group(idx)
        test_group = test_groups.get_group(idx) if idx in test_groups.groups else pd.DataFrame()
        
        # Sample with replacement if needed
        train_replace = len(train_group) < max_train_per_class
        test_replace = len(test_group) < max_test_per_class
        
        sampled_train = train_group.sample(
            n=min(len(train_group), max_train_per_class),
            replace=train_replace,
            random_state=42
        )
        
        if not test_group.empty:
            sampled_test = test_group.sample(
                n=min(len(test_group), max_test_per_class),
                replace=test_replace,
                random_state=42
            )
        else:
            sampled_test = pd.DataFrame(columns=test_df.columns)
        
        sampled_train_dfs.append(sampled_train)
        sampled_test_dfs.append(sampled_test)
        
        print(f"{idx:5d} | {label_name:10s} | {len(train_group):12d} | "
              f"{len(test_group):11d} | {len(sampled_train):10d} | {len(sampled_test):9d}")
    
    # Concatenate all sampled dataframes
    final_train_df = pd.concat(sampled_train_dfs, ignore_index=True)
    final_test_df = pd.concat(sampled_test_dfs, ignore_index=True)
    
    # Convert back to datasets while preserving the original features
    final_train_dataset = Dataset.from_pandas(final_train_df, features=original_features)
    final_test_dataset = Dataset.from_pandas(final_test_df, features=original_features)
    
    # Create new DatasetDict
    sampled_dataset = DatasetDict({
        'train': final_train_dataset,
        'test': final_test_dataset
    })
    
    print("\nFinal dataset sizes:")
    print(f"Train: {len(final_train_dataset)} samples")
    print(f"Test: {len(final_test_dataset)} samples")
    
    # Verify feature structure is preserved
    print("\nVerifying feature structure:")
    print(sampled_dataset['train'].features)
    
    return sampled_dataset

def format_data(examples, tokenizer, text_tag, lab, dataset):
    """Format the data for emotion classification task."""
    formatted_texts = []
    for text, label in zip(examples[text_tag], examples[lab]):
        # Tokenize and decode to normalize text
        tok_text = tokenizer.encode(text, max_length=400, truncation=True)
        text = tokenizer.decode(tok_text)
        
        # Convert label to string
        label_str = dataset['train'].features[lab].int2str(label)
        
        # Format the text with separator token
        formatted_text = f"Classify emotion: {text}{tokenizer.sep_token}"
        formatted_texts.append(formatted_text)
    return {'formatted_text': formatted_texts}

def tokenize_and_prepare(examples, tokenizer):
    """Tokenize examples and prepare them for training."""
    # Tokenize with batch processing
    tokenized = tokenizer(
        examples['formatted_text'],
        padding='max_length',
        max_length=408,
        truncation=True,
        return_tensors="pt"
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

def prepare_tokenizer(tokenizer, dataset, lab):
    """Prepare the tokenizer by adding special tokens and emotion label tokens."""
    # Add emotion labels as tokens
    new_tokens = []
    label2text = dataset['train'].features[lab].names

    for label in label2text:
        special_token = f'{label}'
        
        # Check if the label is already a single token in the tokenizer
        label_tokens = tokenizer.encode(label, add_special_tokens=False)
        is_single_token = len(label_tokens) == 1
        
        if is_single_token:
            print(f"'{label}' is already a single token (ID: {label_tokens[0]})")
        
        # Add to new tokens list
        new_tokens.append(special_token)

    # Add the tokens to the tokenizer
    num_added_tokens = tokenizer.add_tokens(new_tokens)
    print(f"\nAdded {num_added_tokens} new tokens to the tokenizer")

    # Add special tokens
    special_tokens = {
        'pad_token': '<|pad|>',
        'sep_token': '<|sep|>',
        'eos_token': '<|eos|>'
    }
    tokenizer.add_special_tokens(special_tokens)
    
    return tokenizer

def load_model(tokenizer, layer, weights_path):
    """Load and prepare the GPT-2 model."""
    # Load pre-trained GPT-2 model
    base_model = gt.from_pretrained('gpt2')
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.config.m_layer = layer
    
    # Create custom model with masking capabilities
    model = GPT2LMHeadModel(base_model.config)
    model.load_state_dict(torch.load(weights_path))
    
    return model

def evaluate_masking(model, tokenized_dataset, recording_dataset, tokenizer, lab, num_classes, per=0.01):
    """Evaluate the model with masking techniques."""
    # Initial activations recording
    all_fc_vals = []
    base_accuracies = []
    base_confidences = []
    base_comp_acc = []
    base_comp_conf = []
    
    print("Recording activations...")
    for j in range(num_classes):
        # Filter datasets for current class
        dataset_recording = recording_dataset.filter(lambda x: x[lab] in [j])
        dataset = tokenized_dataset.filter(lambda x: x[lab] in [j])
        dataset_complement = tokenized_dataset.filter(lambda x: x[lab] not in [j])
        
        # Record activations
        fc_vals = evaluate_gpt2_classification(lab, model, dataset_recording, tokenizer)
        fc_vals = fc_vals[2]
        all_fc_vals.append(np.array(fc_vals))
        
        # Evaluate base accuracy on class
        acc = evaluate_gpt2_classification(lab, model, dataset, tokenizer)
        base_accuracies.append(acc[0])
        base_confidences.append(acc[1])
        print(f"Class {j} base accuracy: {acc[0]}, confidence: {acc[1]}")
        
        # Evaluate base accuracy on class complement
        acc = evaluate_gpt2_classification(lab, model, dataset_complement, tokenizer)
        base_comp_acc.append(acc[0])
        base_comp_conf.append(acc[1])
        print(f"Class {j} complement base accuracy: {acc[0]}, confidence: {acc[1]}")
    
    # Fit histogram for masking
    print(f"Applying masking with percentage: {per}")
    model.transformer.mask_layer.fit_histogram(all_fc_vals, threshold=per, num_bins=1000)
    
    # Create results table
    results_table = PrettyTable()
    results_table.field_names = [
        "Class", "Base Accuracy", "Base Confidence", 
        "Base Complement Acc", "Base Compliment Conf",
        "STD Accuracy", "STD Confidence", 
        "STD compliment ACC", "STD compliment Conf"
    ]
    
    # Evaluate after masking
    std_accuracies = []
    std_confidences = []
    std_comp_acc = []
    std_comp_conf = []
    class_labels = []
    
    for j in range(num_classes):
        # Set class for masking
        model.transformer.mask_layer.set_class(j)
        
        # Filter datasets
        dataset = tokenized_dataset.filter(lambda x: x[lab] in [j])
        dataset_complement = tokenized_dataset.filter(lambda x: x[lab] not in [j])
        
        class_labels.append(f"Class {j}")
        
        # Evaluate base model
        print(f"Class {j} base accuracy: {base_accuracies[j]}, confidence: {base_confidences[j]}")
        print(f"Class {j} complement base accuracy: {base_comp_acc[j]}, confidence: {base_comp_conf[j]}")
        
        # Evaluate with masking
        acc = evaluate_gpt2_classification(lab, model, dataset, tokenizer) 
        print(f"Accuracy after masking STD: {acc[0]}, confidence: {acc[1]}")
        std_accuracies.append(acc[0])
        std_confidences.append(acc[1])
        
        # Evaluate complement with masking
        acc = evaluate_gpt2_classification(lab, model, dataset_complement, tokenizer)
        print(f"Accuracy after masking STD on complement: {acc[0]}, confidence: {acc[1]}")
        std_comp_acc.append(acc[0])
        std_comp_conf.append(acc[1])
        
        # Add results to table
        results_table.add_row([
            class_labels[j],
            base_accuracies[j],
            base_confidences[j],
            base_comp_acc[j],
            base_comp_conf[j],
            std_accuracies[j],
            std_confidences[j],
            std_comp_acc[j],
            std_comp_conf[j],
        ])
    
    print(results_table)
    return results_table

def main():
    # Configuration
    dataset_name = "dair-ai/emotion"
    text_tag = "text"
    lab = "label"
    layer = 11
    num_classes = 6
    per = 0.01  # Threshold percentage for masking
    
    # Set random seed
    set_seed(42)
    
    # Load dataset
    dataset = load_dataset(dataset_name)
    print(dataset)
    
    # Filter dataset to remove -1 labels
    dataset = dataset.filter(lambda x: x[lab] != -1)
    
    # Optional: Create balanced dataset
    # dataset = sample_balanced_dataset(dataset, max_train_per_class=800, max_test_per_class=200)
    
    # Load and prepare tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer = prepare_tokenizer(tokenizer, dataset, lab)
    
    # Process dataset
    formatted_dataset = dataset.map(
        lambda examples: format_data(examples, tokenizer, text_tag, lab, dataset), 
        batched=True
    )
    
    tokenized_dataset = formatted_dataset.map(
        lambda examples: tokenize_and_prepare(examples, tokenizer), 
        batched=True,
    )
    
    # Prepare model and weights path
    base_path = os.path.join("model_weights", dataset_name)
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    weights_path = os.path.join(base_path, "weights.pth")
    
    # Load model
    model = load_model(tokenizer, layer, weights_path)
    
    # Prepare datasets for evaluation
    tokenized_dataset1 = tokenized_dataset['test']
    recording_dataset = tokenized_dataset['train']
    
    # Run evaluation with masking
    tables = []
    results_table = evaluate_masking(
        model, tokenized_dataset1, recording_dataset, 
        tokenizer, lab, num_classes, per
    )
    tables.append(results_table)
    
    # Display results
    per = 0
    for table in tables:
        per += 0.01
        print(f"percentage: {per}")
        print(table)

if __name__ == "__main__":
    main()