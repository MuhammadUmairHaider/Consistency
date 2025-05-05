#!/usr/bin/env python
"""
Full experiment script: per‑class masking study on Llama‑3.2‑3B with
– classification accuracy & confidence (original logic)
– language‑model perplexity on held‑out corpus
– SQuAD‑v2 QA metrics (F1, EM)

Author: (your name)
Date: 2025‑05‑03
"""

# ───────────────────────── Imports ──────────────────────────
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from utilities import (
    evaluate_llma_classification_batch as evaluate_llma_classification,
    evaluate_llma_language_modeling,
    evaluate_llma_squad,
    mask_range_llma,
    compute_masks,
    reset_llma,
)
from huggingface_hub import login
from prettytable import PrettyTable
import torch, random, numpy as np, os, warnings, matplotlib.pyplot as plt
from collections import Counter

# ─────────────────── HF authentication (optional) ───────────
login("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")

# ───────────────────── Experiment constants ────────────────
SEED          = 42
BATCH_SIZE    = 256           # for perplexity calc
MASK_LAYER    = 5             # not used directly here but retained
TAO_RANGE     = 2.5
PERCENT       = 0.3
NUM_CLASSES   = 14
COMPLEMENT    = True          # evaluate complement sets

# ───────────────────── RNG seeding ─────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.autograd.set_detect_anomaly(True)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.tensor")

# ───────────────────── Utility: class distribution ─────────

def print_class_distribution(labels):
    counts = Counter(labels)
    total  = sum(counts.values())
    print("\nClass Distribution\n" + "-"*40)
    for k, v in counts.items():
        print(f"{k}: {v}  ({v/total*100:.2f}%)")
    plt.bar(counts.keys(), counts.values())
    plt.title("Dataset Class Distribution")
    plt.show()

# ───────────────────── Data loading ────────────────────────
print("Loading custom correct‑predictions dataset …")
correct_ds = load_dataset(
    "json",
    data_files="llama_correct_datasets/correct_predictions_DB_14.json",
)["train"].shuffle(SEED)

# correct_ds = correct_ds.select(range(400))
print("Total examples:", len(correct_ds))
print_class_distribution(correct_ds["label"])

splits   = correct_ds.train_test_split(test_size=0.20, seed=SEED)
train_ds = splits["train"]
test_ds  = splits["test"]

# ──────────────────────────────────────────────────────────────────────────────
# 4. AUXILIARY DATASETS (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
wiki_dataset  = load_dataset("wikipedia", "20220301.en", split="train[:2000]")
squad_dataset = load_dataset("squad", split="validation")
wiki_sample   = wiki_dataset.select(range(2000))
squad_sample  = squad_dataset.select(range(2000))
print("Auxiliary datasets loaded (Wiki + SQuAD).")

# ───────────────────── Model & tokenizer ───────────────────
from models.lama import LlamaForCausalLM   # custom class that supports masking
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Llama-3.2-3B",
                pad_token_id=tokenizer.eos_token_id
            )
base_model.config.m_layer = 27  # layer we’ll mask

weights_path = os.path.join("model_weights", "llama-summarization", "weights.pth")
model = LlamaForCausalLM(base_model.config)
model.load_state_dict(torch.load(weights_path))
model.to("cuda")
print("Model weights loaded.")



# ───────────────────── Table setup ─────────────────────────
results_table = PrettyTable()
results_table.field_names = [
    "Class",
    "Base Acc", "Base Conf", "Base ¬Acc", "Base ¬Conf",
    "Range Acc", "Range Conf", "Range ¬Acc", "Range ¬Conf",
    "Max Acc",  "Max Conf",  "Max ¬Acc",  "Max ¬Conf",
    "Base PPL", "Range PPL", "Max PPL",
    "Base F1",  "Range F1",  "Max F1",
    "Base EM",  "Range EM",  "Max EM",
]

b_ppl                     = evaluate_llma_language_modeling(model, wiki_sample, tokenizer, 2000)
squad_scores              = evaluate_llma_squad(model, squad_sample, tokenizer, 2000)
b_f1, b_em                = squad_scores["f1"], squad_scores["exact_match"]
# ───────────────────── Per‑class experiment ────────────────
for cls in range(NUM_CLASSES):
    print("\n=== Class", cls, "===")
    # reset to fresh weights each iteration
    model = reset_llma(model)

    ds_pos = test_ds.filter(lambda x: x["label"] == cls)
    ds_neg = test_ds.filter(lambda x: x["label"] != cls)
    ds_rec = train_ds.filter(lambda x: x["label"] == cls)

    # ─── Baseline ───────────────────────────────────────────
    b_acc, b_conf, fc_vals = evaluate_llma_classification(model, ds_pos, tokenizer)
    # print(fc_vals)
    # print(len(fc_vals))
    # print(fc_vals[0].shape)
    
    if COMPLEMENT:
        b_neg_acc, b_neg_conf = evaluate_llma_classification(model, ds_neg, tokenizer)[:2]
        
        
    # Recording activations
    
    print("Recording activations")
    
    b_acc, b_conf, fc_vals = evaluate_llma_classification(model, ds_rec, tokenizer)
    
    # print(fc_vals)
    # print(len(fc_vals))
    # print(fc_vals[0].shape)

    # ─── Range‑mask ─────────────────────────────────────────
    
    print("Range masking")
    mask_max, *_ = compute_masks(fc_vals, PERCENT)
    model_rng = mask_range_llma(reset_llma(model), mask_max, fc_vals, TAO_RANGE)

    r_acc, r_conf = evaluate_llma_classification(model_rng, ds_pos, tokenizer)[:2]
    print("Range accuracy:", r_acc, r_conf)
    r_ppl         = evaluate_llma_language_modeling(model_rng, wiki_sample, tokenizer, 5000)
    r_f1, r_em    = evaluate_llma_squad(model_rng, squad_sample, tokenizer, 5000)
    if COMPLEMENT:
        r_neg_acc, r_neg_conf = evaluate_llma_classification(model_rng, ds_neg, tokenizer)[:2]
        print("Range complement accuracy:", r_neg_acc, r_neg_conf)

    # ─── Max‑mask ───────────────────────────────────────────
    
    print("Max masking")
    model_max = mask_range_llma(reset_llma(model), mask_max, fc_vals, torch.inf)

    m_acc, m_conf = evaluate_llma_classification(model_max, ds_pos, tokenizer)[:2]
    print("Max accuracy:", m_acc, m_conf)
    m_ppl         = evaluate_llma_language_modeling(model_max, wiki_sample, tokenizer, 5000)
    m_f1, m_em    = evaluate_llma_squad(model_max, squad_sample, tokenizer, 5000)
    if COMPLEMENT:
        m_neg_acc, m_neg_conf = evaluate_llma_classification(model_max, ds_neg, tokenizer)[:2]
        print("Max complement accuracy:", m_neg_acc, m_neg_conf)

    # ─── Add to table ──────────────────────────────────────
    results_table.add_row([
        f"Class {cls}",
        b_acc, b_conf, b_neg_acc, b_neg_conf,
        r_acc, r_conf, r_neg_acc, r_neg_conf,
        m_acc, m_conf, m_neg_acc, m_neg_conf,
        b_ppl, r_ppl, m_ppl,
        b_f1,  r_f1,  m_f1,
        b_em,  r_em,  m_em,
    ])

# ───────────────────── Print summary ───────────────────────
print("\n===== RESULTS =====")
print(results_table)

print("\nLayer:", MASK_LAYER)
print("Average Base Accuracy:", round(results_table.get_column("Base Acc").avg(), 4))
print("Average Range Accuracy:", round(results_table.get_column("Range Acc").avg(), 4))
print("Average Max Accuracy:",   round(results_table.get_column("Max Acc").avg(), 4))
print("Average Base PPL:", round(results_table.get_column("Base PPL").avg(), 4))
print("Average Range PPL:", round(results_table.get_column("Range PPL").avg(), 4))
print("Average Max PPL:",   round(results_table.get_column("Max PPL").avg(), 4))
print("Average Base F1:", round(results_table.get_column("Base F1").avg(), 4))
print("Average Range F1:", round(results_table.get_column("Range F1").avg(), 4))
print("Average Max F1:",   round(results_table.get_column("Max F1").avg(), 4))
print("Average Base EM:", round(results_table.get_column("Base EM").avg(), 4))
print("Average Range EM:", round(results_table.get_column("Range EM").avg(), 4))
print("Average Max EM:",   round(results_table.get_column("Max EM").avg(), 4))

# ───────────────────── Save results ───────────────────────
results_path = os.path.join("results", "llama_correct_predictions_results.txt")
with open(results_path, "w") as f:
    f.write(str(results_table))
    f.write("\n\n")
    f.write("Layer: " + str(MASK_LAYER) + "\n")
    f.write("Average Base Accuracy: " + str(round(results_table.get_column("Base Acc").avg(), 4)) + "\n")
    f.write("Average Range Accuracy: " + str(round(results_table.get_column("Range Acc").avg(), 4)) + "\n")
    f.write("Average Max Accuracy: " + str(round(results_table.get_column("Max Acc").avg(), 4)) + "\n")
    f.write("Average Base PPL: " + str(round(results_table.get_column("Base PPL").avg(), 4)) + "\n")
    f.write("Average Range PPL: " + str(round(results_table.get_column("Range PPL").avg(), 4)) + "\n")
    f.write("Average Max PPL: " + str(round(results_table.get_column("Max PPL").avg(), 4)) + "\n")
    f.write("Average Base F1: " + str(round(results_table.get_column("Base F1").avg(), 4)) + "\n")
    f.write("Average Range F1: " + str(round(results_table.get_column("Range F1").avg(), 4)) + "\n")
    f.write("Average Max F1: " + str(round(results_table.get_column("Max F1").avg(), 4)) + "\n")
    f.write("Average Base EM: " + str(round(results_table.get_column("Base EM").avg(), 4)) + "\n")
    f.write("Average Range EM: " + str(round(results_table.get_column("Range EM").avg(), 4)) + "\n")
    f.write("Average Max EM: " + str(round(results_table.get_column("Max EM").avg(), 4)) + "\n")
    f.write("\n\n")