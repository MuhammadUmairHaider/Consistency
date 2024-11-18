from datasets import load_dataset
from transformers import GPT2Tokenizer, DataCollatorForLanguageModeling
import random
import numpy as np
from utilities import evaluate_gpt2_classification as evaluate_gpt2_classification, mask_range_gpt,compute_masks, reset_gpt
import torch  

dataset_name = "fancyzhx/ag_news"

text_tag = "text"

# Load dataset and tokenizer
dataset = load_dataset(dataset_name)

# #drop unsupervised data
# from datasets import DatasetDict
# dataset = DatasetDict({k: v for k, v in dataset.items() if k in ['train', 'test']})
# print(dataset)


layer = 0

per = 0.5
num_classes = 150

# tao = 2.5

lab = "label"
tao = torch.inf


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
label2text = dataset['train'].features[lab].names

for label in label2text:
    # Create special token format (with and without space)
    special_token = f'{label}'
    
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
    for text, label in zip(examples[text_tag], examples[lab]):
        # Convert label to string
        
        tok_text = tokenizer.encode(text, max_length=400, truncation=True)
        text = tokenizer.decode(tok_text)
        label_str = dataset['train'].features[lab].int2str(label)
        formatted_text = f"Classify emotion: {text}{tokenizer.sep_token}"#{label_str}{tokenizer.eos_token}"
        formatted_texts.append(formatted_text)
    return {'formatted_text': formatted_texts}

def tokenize_and_prepare(examples):

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
# Process the dataset
formatted_dataset = dataset.map(format_data, batched=True)
tokenized_dataset = formatted_dataset.map(
    tokenize_and_prepare, 
    batched=True,
)

from transformers import GPT2LMHeadModel as gt
from models.gpt2 import GPT2LMHeadModel
# Load pre-trained GPT-2 model
model1 = gt.from_pretrained('gpt2')

model1.resize_token_embeddings(len(tokenizer))

model1.config.m_layer = layer
import os

base_path = os.path.join("model_weights", dataset_name)
if not os.path.exists(base_path):
    os.makedirs(base_path)

weights_path = os.path.join(base_path, "weights.pth")

model = GPT2LMHeadModel(model1.config)


model.load_state_dict(torch.load(weights_path))




from prettytable import PrettyTable
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.tensor")

for mask_layer in range(1,12):

    batch_size = 256
    mask_layer = 5
    compliment = True
    results_table = PrettyTable()
    if(compliment):
        results_table.field_names = results_table.field_names = ["Class", "Base Accuracy", "Base Confidence", "Base Complement Acc", "Base Compliment Conf", "MAX Accuracy", "MAX Confidence", "Max compliment acc", "Max compliment conf"]#, "Same as Max"]#"MAX Accuracy", "MAX Confidence", "Max compliment acc", "Max compliment conf"

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

    tokenized_dataset1 = tokenized_dataset['test']#.shuffle().select(range(200))
    recording_dataset = tokenized_dataset['train']#.shuffle().select(range(200))
    for j in range(0,num_classes):
        model = reset_gpt(model)
        dataset = tokenized_dataset1.filter(lambda x: x[lab] in [j])
        dataset_recording = recording_dataset.filter(lambda x: x[lab] in [j])
        dataset_complement = tokenized_dataset1.filter(lambda x: x[lab] not in [j])
        

        class_labels.append(f"Class {j}")
        acc = evaluate_gpt2_classification(lab, model, dataset, tokenizer)
        print("Class ",j, "base accuracy: ", acc[0], acc[1])
        base_accuracies.append(acc[0])
        base_confidences.append(acc[1])
        if(compliment):
            acc = evaluate_gpt2_classification(lab, model, dataset_complement, tokenizer)
            print("Class ",j, "complement base accuracy: ", acc[0], acc[1])
            base_comp_acc.append(acc[0])
            base_comp_conf.append(acc[1])
            
        print("Recording activations...")
        fc_vals = evaluate_gpt2_classification(lab, model, dataset_recording, tokenizer)
        fc_vals = fc_vals[2]

            
        mask_max, mask_std, mask_intersection, mask_max_low_std, mask_max_high_std, mask_std_high_max = compute_masks(fc_vals,per)
        mask_std = mask_max_low_std
        print("Masking MAX...")
        model = mask_range_gpt(model, mask_max, fc_vals, tao)
        t = int(mask_max.shape[0]-torch.count_nonzero(mask_max))
        print("Total Masked :", t)
        acc = evaluate_gpt2_classification(lab, model, dataset, tokenizer)
        print("accuracy after masking MAX: ", acc[0], acc[1])
        max_accuracies.append(acc[0])
        max_confidences.append(acc[1])
        acc = evaluate_gpt2_classification(lab, model, dataset_complement, tokenizer)
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
            
    print("Layer ", mask_layer)
    print(results_table)
    print("Layer ", mask_layer)
    print("Average Base Accuracy: ",round(sum(base_accuracies)/len(base_accuracies), 4))
    print("Average Base Confidence: ", round(sum(base_confidences)/len(base_confidences), 4))
    print("Average MAX Accuracy: ", round(sum(max_accuracies)/len(max_accuracies), 4))
    print("Average MAX Confidence: ", round(sum(max_confidences)/len(max_confidences), 4))
    print("Average MAX Complement Accuracy: ", round(sum(max_comp_acc)/len(max_comp_acc), 4))
    print("Average MAX Complement Confidence: ", round(sum(max_comp_conf)/len(max_comp_conf), 4))