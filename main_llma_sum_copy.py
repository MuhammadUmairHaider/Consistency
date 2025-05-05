# ──────────────────────────────────────────────────────────────────────────────
# 1. Imports & helpers
# ──────────────────────────────────────────────────────────────────────────────
from datasets        import load_dataset
from transformers    import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from tqdm            import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from prettytable     import PrettyTable
import torch, random, numpy as np, os, warnings, re
from collections     import Counter

from utilities import (
    mask_range_llma, compute_masks, reset_llma,
    evaluate_llma_language_modeling, evaluate_llma_squad, manual_generate_llma
)

# ──────────────────────────────────────────────────────────────────────────────
# 2. HF login & deterministic seed
# ──────────────────────────────────────────────────────────────────────────────
login("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")

seed_value = 42
random.seed(seed_value); np.random.seed(seed_value)
torch.manual_seed(seed_value); torch.cuda.manual_seed_all(seed_value)
torch.autograd.set_detect_anomaly(True)

# ──────────────────────────────────────────────────────────────────────────────
# 3. MAIN‑TASK DATASET  → MNLI  (validation‑matched split, 50 examples)
# ──────────────────────────────────────────────────────────────────────────────
mnli_dataset = load_dataset("glue", "mnli")




mnli = load_dataset("glue", "mnli")

# In GLUE’s label scheme: 0 = entailment, 1 = neutral, 2 = contradiction
ENTAILMENT_ID = 0

mnli_entailment = {
    split: ds.filter(lambda ex: ex["label"] == ENTAILMENT_ID)
    for split, ds in mnli.items()
}


dataset_sample  = mnli_entailment["validation_matched"].select(range(500))  # used for eval
dataset_record  = mnli_entailment["train"].select(range(500))               # for activations
print("MNLI dataset loaded.")

# ──────────────────────────────────────────────────────────────────────────────
# 4. AUXILIARY DATASETS (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
wiki_dataset  = load_dataset("wikipedia", "20220301.en", split="train[:2000]")
squad_dataset = load_dataset("squad", split="validation")
wiki_sample   = wiki_dataset.select(range(200))
squad_sample  = squad_dataset.select(range(200))
print("Auxiliary datasets loaded (Wiki + SQuAD).")

# ──────────────────────────────────────────────────────────────────────────────
# 5. MODEL
# ──────────────────────────────────────────────────────────────────────────────
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
print("Model weights loaded.")

# ──────────────────────────────────────────────────────────────────────────────
# 6. NLI EVALUATION FUNCTION  (replaces summarization)
# ──────────────────────────────────────────────────────────────────────────────
FEW_SHOT = """
Example 1
Premise: A man in a black jacket is riding a bicycle down a city street.
Hypothesis: Someone is cycling in an urban area.
Answer: entailment

Example 2
Premise: A dog is sleeping under a table.
Hypothesis: The animal is wide awake waiting for food.
Answer: contradiction

Example 3
Premise: Two children are playing in a sprinkler on a sunny day.
Hypothesis: Kids are getting wet outside.
Answer: entailment

Now decide whether the hypothesis is entailment, neutral, or contradiction.

Premise: {premise}
Hypothesis: {hypothesis}
Answer:
""".strip()

def evaluate_llma_nli(model, eval_dataset, tok, max_samples=5000):
    if len(eval_dataset) > max_samples:
        eval_dataset = eval_dataset.select(range(max_samples))

    gold, pred, fc_vals = [], [], []
    model.to("cuda").eval()
    id2lbl = {0: "entailment", 1: "neutral", 2: "contradiction"}
    bar    = tqdm(range(len(eval_dataset)), desc="Processing")

    for i in bar:
        ex         = eval_dataset[i]
        prompt     = FEW_SHOT.format(premise=ex["premise"], hypothesis=ex["hypothesis"])
        inputs     = tok([prompt, prompt], return_tensors="pt")
        prompt_len = inputs["input_ids"].shape[1]

        out  = manual_generate_llma(model,
                                    inputs["input_ids"].to("cuda"),
                                    inputs["attention_mask"].to("cuda"),
                                    max_length=inputs['input_ids'].shape[1] +5)

        first_token_id = out[0][0][prompt_len].item()
        # print(tok.decode(out[0][0]))
        first_word     = tok.decode(first_token_id).lower().strip()
        if   first_word.startswith("entail"): pred_lbl = "entailment"
        elif first_word.startswith("contrad"): pred_lbl = "contradiction"
        elif first_word.startswith("neut"): pred_lbl = "neutral"
        else:
            pred_lbl = ""

        gold_lbl = id2lbl.get(ex["label"], ex["label"])
        gold.append(gold_lbl); pred.append(pred_lbl)

        # fc vector at class‑token position (store only if correct)
        if pred_lbl == gold_lbl:
            fc_vals.append(out[2][0][0].squeeze())

        bar.set_description(f"ACC: {accuracy_score(gold,pred):.4f}")

    acc = accuracy_score(gold, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
                           gold, pred, average="macro", zero_division=0)

    return {"accuracy": acc,
            "macro_precision": prec,
            "macro_recall": rec,
            "macro_f1": f1}, fc_vals
# ──────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────
# 7. TABLE SET‑UP  (main task now uses macro‑F1)
# ──────────────────────────────────────────────────────────────────────────────
main_results = PrettyTable()
main_results.field_names = ["Masking Type", "Acc", "Macro‑P", "Macro‑R", "Macro‑F1"]

aux_results  = PrettyTable()
aux_results.field_names  = ["Masking Type", "Perplexity", "Δ PPL %"]

squad_results = PrettyTable()
squad_results.field_names = ["Masking Type", "Exact Match", "F1", "Δ EM", "Δ F1"]

mask_layer = 27
percent    = 0.3

warnings.filterwarnings("ignore", category=UserWarning, module="torch.tensor")

# ──────────────────────────────────────────────────────────────────────────────
# 8. BASELINE EVALUATIONS
# ──────────────────────────────────────────────────────────────────────────────
print("Evaluating baseline on MNLI…")
model = reset_llma(model)
base_scores, _ = evaluate_llma_nli(model, dataset_sample, tokenizer, max_samples=2000)
main_results.add_row(["Base (No Mask)", f"{base_scores['accuracy']:.4f}",
                      f"{base_scores['macro_precision']:.4f}",
                      f"{base_scores['macro_recall']:.4f}",
                      f"{base_scores['macro_f1']:.4f}"])

print("Evaluating baseline on Wiki LM…")
base_lm = evaluate_llma_language_modeling(model, wiki_sample, tokenizer, max_samples=2000)
aux_results.add_row(["Base (No Mask)", f"{base_lm['perplexity']:.4f}", "0%"])

print("Evaluating baseline on SQuAD…")
base_qa = evaluate_llma_squad(model, squad_sample, tokenizer, max_samples=500)
squad_results.add_row(["Base (No Mask)", f"{base_qa['exact_match']:.4f}",
                       f"{base_qa['f1']:.4f}", "0", "0"])

# ──────────────────────────────────────────────────────────────────────────────
# 9. RECORD ACTIVATIONS → MASKS
# ──────────────────────────────────────────────────────────────────────────────
print("Recording activations (train slice)…")
_, fc_vals = evaluate_llma_nli(model, dataset_record, tokenizer, max_samples=2000)
masks = compute_masks(fc_vals, percent)
(mask_max, mask_std, mask_int,
 mask_max_low_std, mask_max_high_std,
 mask_std_high_max, mask_max_random_off,
 mask_random) = masks

# ──────────────────────────────────────────────────────────────────────────────
# 10. MASKING EXPERIMENTS  (MAX and Range)  – metrics use macro‑F1
# ──────────────────────────────────────────────────────────────────────────────
def run_mask(name, tao):
    print(f"\nApplying {name} masking…")
    mdl = reset_llma(model)
    mdl = mask_range_llma(mdl, mask_max, fc_vals, tao)
    masked = int(mask_max.shape[0] - torch.count_nonzero(mask_max))
    print(f"Masked units: {masked}")

    nli_scores, _ = evaluate_llma_nli(mdl, dataset_sample, tokenizer, max_samples=2000)
    lm_scores     = evaluate_llma_language_modeling(mdl, wiki_sample, tokenizer, 2000)
    qa_scores     = evaluate_llma_squad(mdl, squad_sample, tokenizer, 2000)

    # add to tables
    main_results.add_row([
        name, f"{nli_scores['accuracy']:.4f}",
        f"{nli_scores['macro_precision']:.4f}",
        f"{nli_scores['macro_recall']:.4f}",
        f"{nli_scores['macro_f1']:.4f}"
    ])

    lm_pct = ((lm_scores["perplexity"] - base_lm["perplexity"])
              / base_lm["perplexity"]) * 100
    aux_results.add_row([name, f"{lm_scores['perplexity']:.4f}", f"{lm_pct:.2f}%"])

    em_drop  = base_qa['exact_match'] - qa_scores['exact_match']
    f1_drop  = base_qa['f1'] - qa_scores['f1']
    squad_results.add_row([
        name, f"{qa_scores['exact_match']:.4f}", f"{qa_scores['f1']:.4f}",
        f"{em_drop:.4f}", f"{f1_drop:.4f}"
    ])

    return nli_scores, lm_pct, f1_drop

max_scores, max_ppl_inc, f1_drop_max = run_mask("MAX Mask", tao=torch.inf)
rng_scores, rng_ppl_inc, f1_drop_rng = run_mask("Range Mask", tao=3)

# ──────────────────────────────────────────────────────────────────────────────
# 11. PRINT RESULTS
# ──────────────────────────────────────────────────────────────────────────────
print("\nMain Task (MNLI) Results:")
print(main_results)

print("\nAuxiliary Task – Language Modeling:")
print(aux_results)

print("\nAuxiliary Task – SQuAD QA:")
print(squad_results)

# ──────────────────────────────────────────────────────────────────────────────
# 12. SPECIFICITY / DEGRADATION ANALYSIS  (macro‑F1 drop instead of ROUGE)
# ──────────────────────────────────────────────────────────────────────────────
f1_drop_base_rng = base_scores['macro_f1'] - rng_scores['macro_f1']
f1_drop_base_max = base_scores['macro_f1'] - max_scores['macro_f1']

lm_ratio_rng = rng_ppl_inc / (f1_drop_base_rng*100) if f1_drop_base_rng>0 else float('inf')
lm_ratio_max = max_ppl_inc / (f1_drop_base_max*100) if f1_drop_base_max>0 else float('inf')
qa_ratio_rng = f1_drop_rng / f1_drop_base_rng if f1_drop_base_rng>0 else float('inf')
qa_ratio_max = f1_drop_max / f1_drop_base_max if f1_drop_base_max>0 else float('inf')

print("\nDegradation ratios (lower = more specific):")
print(f"  LM  – Range: {lm_ratio_rng:.4f} | MAX: {lm_ratio_max:.4f}")
print(f"  SQuAD– Range: {qa_ratio_rng:.4f} | MAX: {qa_ratio_max:.4f}")

if np.isfinite(lm_ratio_rng) and np.isfinite(qa_ratio_rng):
    rel_spec = (lm_ratio_max/lm_ratio_rng + qa_ratio_max/qa_ratio_rng)/2
    print(f"\n→ Range masking is ~{rel_spec:.2f}× more specific than MAX on average.")
else:
    print("\n→ Could not compute combined specificity (division by zero).")

# ──────────────────────────────────────────────────────────────────────────────
# 13. SAVE TO FILE  (filenames & text updated to NLI)
# ──────────────────────────────────────────────────────────────────────────────
out_path = "masking_comparison_results_nli.txt"
with open(out_path, "w") as f:
    f.write("Main Task (MNLI NLI):\n" + str(main_results) + "\n\n")
    f.write("Language Modeling:\n"  + str(aux_results)  + "\n\n")
    f.write("SQuAD QA:\n"          + str(squad_results)+ "\n\n")
    f.write("Degradation Ratios:\n")
    f.write(f"LM  – Range: {lm_ratio_rng:.4f}, MAX: {lm_ratio_max:.4f}\n")
    f.write(f"QA  – Range: {qa_ratio_rng:.4f}, MAX: {qa_ratio_max:.4f}\n")
print(f"Results saved to {out_path}")
