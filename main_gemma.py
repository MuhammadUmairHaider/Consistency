from datasets import load_dataset
from transformers import GPT2Tokenizer, DataCollatorForLanguageModeling
import random
import numpy as np
from utilities import evaluate_llma_classification as evaluate_llma_classification, mask_range_llma,compute_masks, reset_llma
import torch  

from huggingface_hub import login
login("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")

# Load the dataset from disk
correct_dataset = load_dataset("json", data_files="/u/amo-d1/grad/mha361/work/probless/Sentiment/llama_correct_datasets/correct_predictions_DB_14.json")

# correct_dataset = load_dataset("fancyzhx/ag_news")

correct_dataset = correct_dataset["train"]
num_classes = 14

tao = 2.5

percent = 0.3
# tao = torch.inf


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
from models.gemma import GemmaForCausalLM
import torch


from transformers import AutoTokenizer, AutoModelForCausalLM

model1 = AutoModelForCausalLM.from_pretrained("google/gemma-2b")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")

model1.config.m_layer = 17
import os

base_path = os.path.join("model_weights", 'gemma')
if not os.path.exists(base_path):
    os.makedirs(base_path)

weights_path = os.path.join(base_path, "weights.pth")

# torch.save(model1.state_dict(), weights_path)

model = GemmaForCausalLM(model1.config)

# torch.save(model1.state_dict(), weights_path)

model.load_state_dict(torch.load(weights_path))

print(model)

from prettytable import PrettyTable
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.tensor")

batch_size = 256
mask_layer = 5
compliment = True
results_table = PrettyTable()
if(compliment):
   results_table.field_names = results_table.field_names = ["Class", "Base Accuracy", "Base Confidence", "Base Complement Acc", "Base Compliment Conf", "Range Accuracy", "Range Confidence", "Range compliment acc", "Range compliment conf", "MAX Accuracy", "MAX Confidence", "Max compliment acc", "Max compliment conf"]#, "Same as Max"]#"MAX Accuracy", "MAX Confidence", "Max compliment acc", "Max compliment conf"

class_labels = []
base_accuracies = []
base_confidences = []
base_comp_acc = []
base_comp_conf = []
std_masked_counts = []
std_accuracies = []
std_confidences = []
std_comp_acc = []
std_comp_conf = []
max_masked_counts = []
max_accuracies = []
max_confidences = []
max_comp_acc = []
max_comp_conf = []
diff_from_max = []
total_masked = []



from collections import Counter
import matplotlib.pyplot as plt

def print_class_distribution(dataset_labels):
    # Count instances of each class
    class_counts = Counter(dataset_labels)
    total = sum(class_counts.values())
    
    # Print distribution
    print("\nClass Distribution:")
    print("-" * 40)
    for class_name, count in class_counts.items():
        percentage = (count / total) * 100
        print(f"{class_name}: {count} instances ({percentage:.2f}%)")
    
    # Optional: Create bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.title("Dataset Class Distribution")
    plt.xlabel("Classes")
    plt.ylabel("Number of Instances")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



correct_dataset = correct_dataset.shuffle(seed=42)

# correct_dataset = correct_dataset.select(range(4000))

print_class_distribution(correct_dataset['label'])




tokenized_dataset = correct_dataset.train_test_split(
    test_size=0.2,  # 20% for test set
    seed=42        # for reproducibility
)
print(tokenized_dataset)

for j in range(0,num_classes):
    model = reset_llma(model)
    dataset = tokenized_dataset['test'].filter(lambda x: x['label'] in [j])
    dataset_complement = tokenized_dataset['test'].filter(lambda x: x['label'] not in [j])
    dataset_record = tokenized_dataset['train'].filter(lambda x: x['label'] in [j])

    class_labels.append(f"Class {j}")
    acc = evaluate_llma_classification(model, dataset, tokenizer)
    print("Class ",j, "base accuracy: ", acc[0], acc[1])
    base_accuracies.append(acc[0])
    base_confidences.append(acc[1])
    if(compliment):
        acc = evaluate_llma_classification(model, dataset_complement, tokenizer)
        print("Class ",j, "complement base accuracy: ", acc[0], acc[1])
        base_comp_acc.append(acc[0])
        base_comp_conf.append(acc[1])
        
    print("Recording activations...")
    fc_vals = evaluate_llma_classification(model, dataset_record, tokenizer)
    fc_vals = fc_vals[2]

        
    mask_max, mask_std, mask_intersection, mask_max_low_std, mask_max_high_std, mask_std_high_max,mask_max_random_off, mask_random = compute_masks(fc_vals,percent)
    tao = 2.5
    mask_std = mask_max_low_std
    print("Masking Range...")
    model = mask_range_llma(model, mask_max, fc_vals, tao)
    t = int(mask_max.shape[0]-torch.count_nonzero(mask_max))
    print("Total Masked :", t)
    acc = evaluate_llma_classification(model, dataset, tokenizer)
    print("accuracy after masking Range: ", acc[0], acc[1])
    std_accuracies.append(acc[0])
    std_confidences.append(acc[1])
    acc = evaluate_llma_classification(model, dataset_complement, tokenizer)
    print("accuracy after masking Range on complement: ", acc[0], acc[1])
    std_comp_acc.append(acc[0])
    std_comp_conf.append(acc[1])
    
    model = reset_llma(model)    
    tao = torch.inf
    mask_std = mask_max_low_std
    print("Masking MAX...")
    model = mask_range_llma(model, mask_max, fc_vals, tao)
    t = int(mask_max.shape[0]-torch.count_nonzero(mask_max))
    print("Total Masked :", t)
    acc = evaluate_llma_classification(model, dataset, tokenizer)
    print("accuracy after masking MAX: ", acc[0], acc[1])
    max_accuracies.append(acc[0])
    max_confidences.append(acc[1])
    acc = evaluate_llma_classification(model, dataset_complement, tokenizer)
    print("accuracy after masking MAX on complement: ", acc[0], acc[1])
    max_comp_acc.append(acc[0])
    max_comp_conf.append(acc[1])
    
    
    
    
    
    
    if(compliment):
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
            max_accuracies[j],
            max_confidences[j],
            max_comp_acc[j],
            max_comp_conf[j]
        ])
print(results_table)
print("Layer ", mask_layer)
print("Average Base Accuracy: ",round(sum(base_accuracies)/len(base_accuracies), 4))
print("Average Base Confidence: ", round(sum(base_confidences)/len(base_confidences), 4))
print("Average MAX Accuracy: ", round(sum(max_accuracies)/len(max_accuracies), 4))
print("Average MAX Confidence: ", round(sum(max_confidences)/len(max_confidences), 4))
print("Average MAX Complement Accuracy: ", round(sum(max_comp_acc)/len(max_comp_acc), 4))
print("Average MAX Complement Confidence: ", round(sum(max_comp_conf)/len(max_comp_conf), 4))