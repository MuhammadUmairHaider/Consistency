from datasets import load_dataset
from transformers import AutoTokenizer
import random
import numpy as np
import torch
from tqdm import tqdm
from rouge_score import rouge_scorer
from utilities import mask_range_llma, compute_masks, reset_llma, evaluate_llma_summarization, evaluate_llma_language_modeling, evaluate_llma_squad
from huggingface_hub import login
import re
from collections import Counter

# Login to Hugging Face (if needed)
login("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")

# Set random seed for reproducibility
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
if torch.cuda.is_available():
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.autograd.set_detect_anomaly(True)

# Load the CNN/Daily Mail dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")
print("Dataset loaded:", dataset)

# Load Wiki dataset for auxiliary task
wiki_dataset = load_dataset("wikipedia", "20220301.en", split="train[:500]")
print("Wiki dataset loaded for auxiliary task")

# Load SQuAD dataset for auxiliary task
squad_dataset = load_dataset("squad", split="validation")
print(f"SQuAD dataset loaded for auxiliary task: {len(squad_dataset)} examples")

# Load model and tokenizer
from models.lama import LlamaForCausalLM
from transformers import AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
tokenizer.pad_token = tokenizer.eos_token
model1 = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B", pad_token_id=tokenizer.eos_token_id)

model1.config.m_layer = 27

import os
# Prepare model directory
base_path = os.path.join("model_weights", 'llama-summarization')
if not os.path.exists(base_path):
    os.makedirs(base_path)
weights_path = os.path.join(base_path, "weights.pth")
# torch.save(model1.state_dict(), weights_path)
model = LlamaForCausalLM(model1.config)
model.load_state_dict(torch.load(weights_path))
print("Model loaded")

# Main experiment
from prettytable import PrettyTable
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.tensor")

# Setup parameters
batch_size = 256
mask_layer = 27
percent = 0.3

# Create tables for all tasks
main_results_table = PrettyTable()
main_results_table.field_names = ["Masking Type", "ROUGE-1", "ROUGE-2", "ROUGE-L"]

aux_results_table = PrettyTable()
aux_results_table.field_names = ["Masking Type", "Perplexity", "Perplexity Increase %"]

squad_results_table = PrettyTable()
squad_results_table.field_names = ["Masking Type", "Exact Match", "F1 Score", "EM Drop", "F1 Drop"]

# Create smaller datasets for efficiency
dataset_sample = dataset['test'].select(range(50))
dataset_record = dataset['train'].select(range(50))  # For recording activations
wiki_sample = wiki_dataset.select(range(200))  # For language modeling task
squad_sample = squad_dataset.select(range(200))  # For question answering task

# Consistency check
print(f"Consistency check - First example in dataset_sample: {dataset_sample[0]['article'][:50]}...")
print(f"Consistency check - First example in wiki_sample: {wiki_sample[0]['text'][:50]}...")
print(f"Consistency check - First example in squad_sample: {squad_sample[0]['question'][:50]}...")

print("Starting base evaluation on main task...")
model = reset_llma(model)
base_scores, *_ = evaluate_llma_summarization(model, dataset_sample, tokenizer, max_samples=100)
main_results_table.add_row([
    "Base (No Masking)",
    f"{base_scores['rouge1']:.4f}",
    f"{base_scores['rouge2']:.4f}",
    f"{base_scores['rougeL']:.4f}"
])

print("Starting base evaluation on language modeling task...")
base_aux_scores = evaluate_llma_language_modeling(model, wiki_sample, tokenizer, max_samples=200)
base_perplexity = base_aux_scores["perplexity"]
aux_results_table.add_row(["Base (No Masking)", f"{base_perplexity:.4f}", "0%"])

print("Starting base evaluation on SQuAD task...")
base_squad_scores = evaluate_llma_squad(model, squad_sample, tokenizer, max_samples=50)
squad_results_table.add_row([
    "Base (No Masking)",
    f"{base_squad_scores['exact_match']:.4f}",
    f"{base_squad_scores['f1']:.4f}",
    "0.0000",
    "0.0000"
])

print("Recording activations...")
*_, fc_vals = evaluate_llma_summarization(model, dataset_record, tokenizer, max_samples=100)

# Compute masks
mask_max, mask_std, mask_intersection, mask_max_low_std, mask_max_high_std, mask_std_high_max, mask_max_random_off, mask_random = compute_masks(fc_vals, percent)

# MAX masking (complete)
print("Masking with MAX...")
model = reset_llma(model)
tao = torch.inf
model = mask_range_llma(model, mask_max_low_std, fc_vals, tao)
t = int(mask_max_low_std.shape[0]-torch.count_nonzero(mask_max_low_std))
print(f"Total Masked: {t}")

# Evaluate MAX masking on main task
max_scores, *_ = evaluate_llma_summarization(model, dataset_sample, tokenizer, max_samples=100)
main_results_table.add_row([
    "MAX Masking",
    f"{max_scores['rouge1']:.4f}",
    f"{max_scores['rouge2']:.4f}",
    f"{max_scores['rougeL']:.4f}"
])

# Evaluate MAX masking on language modeling task
max_aux_scores = evaluate_llma_language_modeling(model, wiki_sample, tokenizer, max_samples=200)
max_perplexity = max_aux_scores["perplexity"]
max_perplexity_increase = ((max_perplexity - base_perplexity) / base_perplexity) * 100
aux_results_table.add_row([
    "MAX Masking", 
    f"{max_perplexity:.4f}", 
    f"{max_perplexity_increase:.2f}%"
])

# Evaluate MAX masking on SQuAD task
max_squad_scores = evaluate_llma_squad(model, squad_sample, tokenizer, max_samples=50)
em_drop_max = base_squad_scores['exact_match'] - max_squad_scores['exact_match']
f1_drop_max = base_squad_scores['f1'] - max_squad_scores['f1']
squad_results_table.add_row([
    "MAX Masking",
    f"{max_squad_scores['exact_match']:.4f}",
    f"{max_squad_scores['f1']:.4f}",
    f"{em_drop_max:.4f}",
    f"{f1_drop_max:.4f}"
])

# Range masking (partial)
print("Masking with Range...")
model = reset_llma(model)
tao = 3
model = mask_range_llma(model, mask_max_low_std, fc_vals, tao)
t = int(mask_max_low_std.shape[0]-torch.count_nonzero(mask_max_low_std))
print(f"Total Masked: {t}")

# Evaluate Range masking on main task
range_scores, *_ = evaluate_llma_summarization(model, dataset_sample, tokenizer, max_samples=100)
main_results_table.add_row([
    "Range Masking",
    f"{range_scores['rouge1']:.4f}",
    f"{range_scores['rouge2']:.4f}",
    f"{range_scores['rougeL']:.4f}"
])

# Evaluate Range masking on language modeling task
range_aux_scores = evaluate_llma_language_modeling(model, wiki_sample, tokenizer, max_samples=2000)
range_perplexity = range_aux_scores["perplexity"]
range_perplexity_increase = ((range_perplexity - base_perplexity) / base_perplexity) * 100
aux_results_table.add_row([
    "Range Masking", 
    f"{range_perplexity:.4f}", 
    f"{range_perplexity_increase:.2f}%"
])

# Evaluate Range masking on SQuAD task
range_squad_scores = evaluate_llma_squad(model, squad_sample, tokenizer, max_samples=50)
em_drop_range = base_squad_scores['exact_match'] - range_squad_scores['exact_match']
f1_drop_range = base_squad_scores['f1'] - range_squad_scores['f1']
squad_results_table.add_row([
    "Range Masking",
    f"{range_squad_scores['exact_match']:.4f}",
    f"{range_squad_scores['f1']:.4f}",
    f"{em_drop_range:.4f}",
    f"{f1_drop_range:.4f}"
])

# Print results for main task
print("\nResults for Main Task (Summarization):")
print(main_results_table)
print(f"Layer: {mask_layer}")
print(f"Base ROUGE-1: {base_scores['rouge1']:.4f}")
print(f"Range Masking ROUGE-1: {range_scores['rouge1']:.4f}")
print(f"MAX Masking ROUGE-1: {max_scores['rouge1']:.4f}")
print(f"ROUGE-1 Drop (Range): {base_scores['rouge1'] - range_scores['rouge1']:.4f}")
print(f"ROUGE-1 Drop (MAX): {base_scores['rouge1'] - max_scores['rouge1']:.4f}")

# Print results for language modeling task
print("\nResults for Auxiliary Task 1 (Language Modeling):")
print(aux_results_table)

# Print results for SQuAD task
print("\nResults for Auxiliary Task 2 (SQuAD Question Answering):")
print(squad_results_table)

# Calculate degradation ratios for both auxiliary tasks
print("\nDegradation Analysis:")
summarization_drop_range = base_scores['rouge1'] - range_scores['rouge1']
summarization_drop_max = base_scores['rouge1'] - max_scores['rouge1']

# Language modeling degradation ratio
if summarization_drop_range > 0.001:
    lm_range_ratio = range_perplexity_increase / (summarization_drop_range * 100)
else:
    lm_range_ratio = float('inf')
    
if summarization_drop_max > 0.001:
    lm_max_ratio = max_perplexity_increase / (summarization_drop_max * 100)
else:
    lm_max_ratio = float('inf')

# SQuAD degradation ratio (using F1 score)
if summarization_drop_range > 0.001 and f1_drop_range > 0.001:
    qa_range_ratio = f1_drop_range / summarization_drop_range
else:
    qa_range_ratio = float('inf')
    
if summarization_drop_max > 0.001 and f1_drop_max > 0.001:
    qa_max_ratio = f1_drop_max / summarization_drop_max
else:
    qa_max_ratio = float('inf')

# Create table for language modeling degradation
lm_degradation_table = PrettyTable()
lm_degradation_table.field_names = [
    "Masking Type", 
    "ROUGE-1 Drop", 
    "Perplexity Increase %", 
    "Degradation Ratio (lower is better)"
]
lm_degradation_table.add_row([
    "Range Masking", 
    f"{summarization_drop_range:.4f}", 
    f"{range_perplexity_increase:.2f}%",
    f"{lm_range_ratio:.4f}"
])
lm_degradation_table.add_row([
    "MAX Masking", 
    f"{summarization_drop_max:.4f}", 
    f"{max_perplexity_increase:.2f}%",
    f"{lm_max_ratio:.4f}"
])

# Create table for SQuAD degradation
qa_degradation_table = PrettyTable()
qa_degradation_table.field_names = [
    "Masking Type", 
    "ROUGE-1 Drop (Main)", 
    "F1 Drop (QA)", 
    "Degradation Ratio (lower is better)"
]
qa_degradation_table.add_row([
    "Range Masking", 
    f"{summarization_drop_range:.4f}", 
    f"{f1_drop_range:.4f}",
    f"{qa_range_ratio:.4f}"
])
qa_degradation_table.add_row([
    "MAX Masking", 
    f"{summarization_drop_max:.4f}", 
    f"{f1_drop_max:.4f}",
    f"{qa_max_ratio:.4f}"
])

print("Language Modeling Degradation Analysis:")
print(lm_degradation_table)

print("\nSQuAD QA Degradation Analysis:")
print(qa_degradation_table)

# Calculate combined specificity score (average of both auxiliary tasks)
lm_relative_specificity = lm_max_ratio / lm_range_ratio if lm_range_ratio > 0 else float('inf')
qa_relative_specificity = qa_max_ratio / qa_range_ratio if qa_range_ratio > 0 else float('inf')

# Average specificity (if both are valid)
if not (np.isinf(lm_relative_specificity) or np.isinf(qa_relative_specificity)):
    combined_specificity = (lm_relative_specificity + qa_relative_specificity) / 2
    specificity_description = f"Range masking is {combined_specificity:.2f}x more specific than MAX masking on average"
else:
    # Use the non-infinite one if available
    if not np.isinf(lm_relative_specificity):
        combined_specificity = lm_relative_specificity
        specificity_description = f"Range masking is {combined_specificity:.2f}x more specific than MAX masking (based on LM task)"
    elif not np.isinf(qa_relative_specificity):
        combined_specificity = qa_relative_specificity
        specificity_description = f"Range masking is {combined_specificity:.2f}x more specific than MAX masking (based on QA task)"
    else:
        combined_specificity = float('inf')
        specificity_description = "Could not determine relative specificity (division by zero)"

print(f"\nCombined Specificity Analysis:")
print(specificity_description)

# Write results to file
with open("masking_comparison_results.txt", "w") as f:
    f.write("Results for Main Task (Summarization):\n")
    f.write(str(main_results_table) + "\n\n")
    
    f.write("Results for Auxiliary Task 1 (Language Modeling):\n")
    f.write(str(aux_results_table) + "\n\n")
    
    f.write("Results for Auxiliary Task 2 (SQuAD Question Answering):\n")
    f.write(str(squad_results_table) + "\n\n")
    
    f.write("Language Modeling Degradation Analysis:\n")
    f.write(str(lm_degradation_table) + "\n\n")
    
    f.write("SQuAD QA Degradation Analysis:\n")
    f.write(str(qa_degradation_table) + "\n\n")
    
    f.write(f"Layer: {mask_layer}\n")
    f.write(f"Masking Percentage: {percent}\n")
    f.write(f"Range Masking Tau: {3.5}\n\n")
    
    f.write("Specificity Analysis:\n")
    f.write(f"LM Specificity (MAX/Range): {lm_relative_specificity:.4f}\n")
    f.write(f"QA Specificity (MAX/Range): {qa_relative_specificity:.4f}\n")
    f.write(f"Combined Specificity: {combined_specificity:.4f}\n")
    f.write(f"{specificity_description}\n")

print("Results saved to masking_comparison_results.txt")