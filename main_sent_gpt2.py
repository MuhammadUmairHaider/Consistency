from datasets import load_dataset
from transformers import GPT2Tokenizer, DataCollatorForLanguageModeling
import random
import numpy as np
from utilities import evaluate_gpt2_classification, mask_range_gpt,compute_masks, evaluate_gpt2_classification_batch
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
# Load dataset
dataset = load_dataset("dair-ai/emotion")

# dataset = dataset.filter(lambda x: x['label'] not in [2])

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
special_tokens = '[Label]'

# Add the special tokens to the tokenizer
tokenizer.add_tokens(special_tokens)
tokenizer.pad_token = tokenizer.eos_token

# Function to format the dataset (without removing original columns)
def format_data(examples):
    formatted_texts = []
    for text, label in zip(examples['text'], examples['label']):
        label_str = dataset['train'].features['label'].int2str(label)  # Convert label to string
        formatted_texts.append(f"{text}[Label] {label_str}")
    return {'formatted_text': formatted_texts}  # Create a new field for the formatted text

# Apply formatting to the dataset
formatted_dataset = dataset.map(format_data, batched=True)

# Tokenize the formatted dataset
def tokenize_function(examples):
    return tokenizer(examples["formatted_text"], padding=True, truncation=True)

tokenized_dataset = formatted_dataset.map(tokenize_function, batched=True)

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Keep the original 'text' and 'label' columns intact
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "text", "label"])

from transformers import GPT2LMHeadModel as gt, Trainer, TrainingArguments
from models.gpt2 import GPT2LMHeadModel
# Load pre-trained GPT-2 model
model1 = gt.from_pretrained('gpt2')

model1.resize_token_embeddings(len(tokenizer))

model1.config.m_layer = 11
import os

base_path = os.path.join("model_weights", 'gpt2-emotion-classification')
if not os.path.exists(base_path):
    os.makedirs(base_path)

weights_path = os.path.join(base_path, "weights.pth")

# torch.save(model1.state_dict(), weights_path)

model = GPT2LMHeadModel(model1.config)


model.load_state_dict(torch.load(weights_path))


# # Use the function
# test_dataset = tokenized_dataset['test']
# accuracy, report, true_labels, predicted_labels, confidence, all_hidden = evaluate_gpt2_classification(model, test_dataset, tokenizer)



# print(f"Accuracy: {accuracy:.4f}")
# print("Classification Report:")
# print("confidence: ", confidence)
# print(report)

# # If you want to see the actual labels and predictions
# print("\nSample of True Labels:", true_labels[:10])
# print("Sample of Predicted Labels:", predicted_labels[:10])

# # Check a few samples of the reconstructed text
# print("\nSample of reconstructed texts:")
# for i in range(5):
#     full_text = tokenizer.decode(test_dataset[i]['input_ids'])
#     print(f"Sample {i}: {full_text}")

# # Print some statistics
# print(f"\nTotal samples processed: {len(true_labels)}")
# print(f"Unique true labels: {set(true_labels)}")
# print(f"Unique predicted labels: {set(predicted_labels)}")




from prettytable import PrettyTable


batch_size = 256
mask_layer = 5
text_tag = "text"
compliment = True
results_table = PrettyTable()
if(compliment):
   results_table.field_names = results_table.field_names = ["Class", "Base Accuracy", "Base Confidence", "Base Complement Acc", "Base Compliment Conf", "MAX Accuracy", "MAX Confidence", "Max compliment acc", "Max compliment conf"]#, "Same as Max"]#"MAX Accuracy", "MAX Confidence", "Max compliment acc", "Max compliment conf"
# results_table.field_names = ["Class", "Base Accuracy", "Base Confidence", "STD Accuracy", "STD Confidence", "Same as Max"]#, "MAX Accuracy", "MAX Confidence", "Max compliment acc", "Max compliment conf"]

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

tokenized_dataset = tokenized_dataset['test'].select(range(200))
# tokenized_dataset = tokenized_dataset[:20]
for j in range(0,6):
    # model = get_model_distilbert("esuriddick/distilbert-base-uncased-finetuned-emotion", mask_layer)
    
    model.load_state_dict(torch.load(weights_path))
    dataset = tokenized_dataset.filter(lambda x: x['label'] in [j])
    dataset_complement = tokenized_dataset.filter(lambda x: x['label'] not in [j])
    
    if(j==6):
        dataset = tokenized_dataset

    class_labels.append(f"Class {j}")
    acc = evaluate_gpt2_classification(model, dataset, tokenizer)
    print("Class ",j, "base accuracy: ", acc[0], acc[1])
    base_accuracies.append(acc[0])
    base_confidences.append(acc[1])
    if(compliment):
        acc = evaluate_gpt2_classification(model, dataset_complement, tokenizer)
        print("Class ",j, "complement base accuracy: ", acc[0], acc[1])
        base_comp_acc.append(acc[0])
        base_comp_conf.append(acc[1])
        
    print("Recording activations...")
    fc_vals = evaluate_gpt2_classification(model, dataset, tokenizer)
    fc_vals = fc_vals[2]

        
    mask_max, mask_std, mask_intersection, mask_max_low_std, mask_max_high_std, mask_std_high_max = compute_masks(fc_vals,0.5)
    mask_std = mask_max_low_std
    print("Masking MAX...")
    # model = mask_distillbert(model,mask_max)
    model = mask_range_gpt(model, mask_max, fc_vals)
    t = int(mask_max.shape[0]-torch.count_nonzero(mask_max))
    print("Total Masked :", t)
    acc = evaluate_gpt2_classification(model, dataset, tokenizer)
    print("accuracy after masking MAX: ", acc[0], acc[1])
    max_accuracies.append(acc[0])
    max_confidences.append(acc[1])
    acc = evaluate_gpt2_classification(model, dataset_complement, tokenizer)
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