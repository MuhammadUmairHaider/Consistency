from datasets import load_dataset
from transformers import GPT2Tokenizer, DataCollatorForLanguageModeling
import random
import numpy as np
from utilities import evaluate_gpt2_classification_batch as evaluate_gpt2_classification, mask_range_gpt,compute_masks, reset_gpt
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
dataset = load_dataset("fancyzhx/dbpedia_14")

dataset = dataset.rename_column("content", "text")

# dataset = dataset.filter(lambda x: x['label'] not in [2])

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
special_tokens = '[Label]'

# Add the special tokens to the tokenizer
tokenizer.add_tokens(special_tokens)
tokenizer.pad_token = tokenizer.eos_token


special_tokens_dict = {}
new_tokens = []
label2text = dataset['train'].features['label'].names

for label in label2text:
    # Create special token format (with and without space)
    special_token = f'[{label}]'
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

# Function to format the dataset (without removing original columns)
def format_data(examples):
    formatted_texts = []
    
    for text, label in zip(examples['text'], examples['label']):
        tok_text = tokenizer.encode(text, max_length=170, truncation=True)
        text = tokenizer.decode(tok_text)
        label_str = dataset['train'].features['label'].int2str(label)  # Convert label to string
        formatted_texts.append(f"{text}[Label][{label_str}]")
    return {'formatted_text': formatted_texts}  # Create a new field for the formatted text

# Apply formatting to the dataset
formatted_dataset = dataset.map(format_data, batched=True)

# Tokenize the formatted dataset
def tokenize_function(examples):
    return tokenizer(examples["formatted_text"], padding='max_length', max_length = 180, truncation=True)

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

base_path = os.path.join("model_weights", 'gpt2-db14-classification')
if not os.path.exists(base_path):
    os.makedirs(base_path)

weights_path = os.path.join(base_path, "weights.pth")

torch.save(model1.state_dict(), weights_path)

model = GPT2LMHeadModel(model1.config)


model.load_state_dict(torch.load(weights_path))


# Training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-emotion-classification",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    logging_dir='./logs',
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'].remove_columns(['label', 'text','formatted_text']),
    eval_dataset=tokenized_dataset['test'].remove_columns(['label', 'text','formatted_text']),
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

torch.save(model.state_dict(), weights_path)




from prettytable import PrettyTable
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.tensor")

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

tokenized_dataset = tokenized_dataset['test']#.shuffle().select(range(200))
# tokenized_dataset = tokenized_dataset[:20]
for j in range(0,14):
    # model = get_model_distilbert("esuriddick/distilbert-base-uncased-finetuned-emotion", mask_layer)
    max=0
    # for i in tokenized_dataset:
    #     print(i['input_ids'].shape)
    #     if(i['input_ids'].shape[0]>max):
    #         max = i['input_ids'].shape[0]
    # print("Max: ", max)
    
    model = reset_gpt(model)
    dataset = tokenized_dataset.filter(lambda x: x['label'] in [j])
    dataset_complement = tokenized_dataset.filter(lambda x: x['label'] not in [j])
    
    if(j==14):
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