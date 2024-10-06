import torch
from tqdm import tqdm
import numpy as np
from models.distilbert import DistilBertForSequenceClassification
from transformers import AutoModelForSequenceClassification
from models.bert import BertForSequenceClassification

from models.distilbert import MaskLayer
import os
torch.manual_seed(0)
np.random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_predictions_diff(a, b):
    return np.sum(np.abs(a - b))

def rank_dataset_by_diff(dataset, in_dataset):
    for i in range(len(dataset)):
        dataset[i] = list(dataset[i])  # Convert tuple to list
        dataset[i][3] = calculate_predictions_diff(dataset[i][2][dataset[i][1]], in_dataset[i][2][dataset[i][1]])  # Modify the value
        # dataset[i] = tuple(dataset[i]) 
        
    sorted_indices = np.argsort([row[3] for row in dataset])
    # Reorder the list using sorted indices
    sorted_dataset = [dataset[i] for i in sorted_indices]
    return sorted_dataset
        

def compute_accuracy(dataset, model, tokenizer, text_tag='sentence', batch_size=32, in_aug_dataset=[]):
    correct = 0
    total_confidence = 0.0
    total_samples = len(dataset)
    progress_bar = tqdm(total=total_samples)
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    augmented_dataset = []
    
    i = 0
    
    texts_all = []
    labels_all = []
    predictions_all = []
    with torch.no_grad():  # Disable gradient computation
        for i in range(0, total_samples, min(batch_size, total_samples-i)):
            batch = dataset[i:min(i+batch_size, total_samples)]
            texts = batch[text_tag]
            labels = torch.tensor(batch['label']).to(device)
            
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs[0], dim=-1)
            predicted_class_idx = torch.argmax(predictions, dim=-1)
            
            batch_correct = (predicted_class_idx == labels)
            correct += batch_correct.sum().item()
            texts_all.extend(texts)
            labels_all.extend(labels.cpu().numpy())
            predictions_all.extend(predictions.cpu().numpy())

            # Sum only the predicted class's probability for correct predictions
            total_confidence += predictions[batch_correct, predicted_class_idx[batch_correct]].sum().item()
            
            progress_bar.update(len(texts))
            
    augmented_dataset = list(zip(texts_all, labels_all, predictions_all, [0]*len(texts_all)))
    
    if(in_aug_dataset!=[]):
        augmented_dataset = rank_dataset_by_diff(augmented_dataset, in_aug_dataset)
        
    # print(i, total_samples)
    progress_bar.close()
    
    accuracy = correct / total_samples
    if(correct == 0):
        average_confidence = 0
    else:
        average_confidence = total_confidence / correct
    
    return round(accuracy, 4), round(average_confidence, 4), augmented_dataset

def record_activations(dataset, model, tokenizer, text_tag='sentence', batch_size=32, mask_layer=0):
    total_samples = len(dataset)
    progress_bar = tqdm(total=total_samples)
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    fc_vals = []
    i = 0
    with torch.no_grad():
        for i in range(0, total_samples, min(batch_size, total_samples-i)):
            batch = dataset[i:min(i+batch_size, total_samples)]
            texts = batch[text_tag]

            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            outputs = model(**inputs)
            
            fc_vals.extend(outputs[1][mask_layer+1][:, 0].squeeze().cpu().numpy())
            progress_bar.update(len(texts))
    progress_bar.close()
        
    return fc_vals
def get_model_distilbert(directory_path, layer):
    base_path = os.path.join("model_weights", directory_path)
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    weights_path = os.path.join(base_path, "weights.pth")

    # model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=4)
    model = AutoModelForSequenceClassification.from_pretrained(directory_path)
    model.config.m_layer = layer
    #save weights
    torch.save(model.state_dict(), weights_path)

    model = DistilBertForSequenceClassification(model.config)
    #load weights
    model.load_state_dict(torch.load(weights_path))
    
    return model

def get_model_bert(directory_path, layer):
    base_path = os.path.join("model_weights", directory_path)
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    weights_path = os.path.join(base_path, "weights.pth")

    model = AutoModelForSequenceClassification.from_pretrained(directory_path)
    model.config.m_layer = layer
    #save weights
    torch.save(model.state_dict(), weights_path)

    model = BertForSequenceClassification(model.config)
    #load weights
    model.load_state_dict(torch.load(weights_path))
    
    return model

import numpy as np
import torch

def compute_masks(fc_vals, percent):
    # Convert input to numpy array
    fc_vals_array = np.array(fc_vals)
    
    # Compute statistics
    mean_vals = np.mean(np.abs(fc_vals_array), axis=0)
    std_vals = np.std(fc_vals_array, axis=0)
    min_vals = np.min(fc_vals_array, axis=0)
    max_vals = np.max(fc_vals_array, axis=0)
    
    # Normalize standard deviation
    std_vals_normalized = (std_vals - min_vals) / (max_vals - min_vals)
    
    # Convert to PyTorch tensors
    mean_vals_tensor = torch.from_numpy(mean_vals)
    std_vals_tensor = torch.from_numpy(std_vals_normalized)
    
    # Compute masks
    mask_max = compute_max_mask(mean_vals_tensor, percent)
    mask_std = compute_std_mask(std_vals_tensor, percent)
    mask_max_low_std = compute_max_low_std_mask(mean_vals_tensor, std_vals_tensor, percent)
    mask_std_high_max = compute_std_high_max_mask(mean_vals_tensor, std_vals_tensor, percent)
    mask_max_high_std = comute_max_high_std_mask(mean_vals_tensor, std_vals_tensor, percent)
    mask_intersection = mask_max * mask_std
    
    return mask_max, mask_std, mask_intersection, mask_max_low_std, mask_max_high_std, mask_std_high_max

def compute_max_mask(values, percent):
    sorted_indices = torch.argsort(values, descending=True)
    mask_count = int(percent * len(values))
    mask = torch.ones_like(values)
    mask[sorted_indices[:mask_count]] = 0.0
    return mask

def compute_std_mask(values, percent):
    sorted_indices = torch.argsort(values, descending=False)
    mask_count = int(percent * len(values))
    mask = torch.ones_like(values)
    mask[sorted_indices[:mask_count]] = 0.0
    return mask

def comute_max_high_std_mask(mean_vals, std_vals, percent):
    
    
    
    bottom_50_percent_max_count = int(0.20 * len(mean_vals))
    bottom_50_percent_max_indices = torch.argsort(mean_vals)[:bottom_50_percent_max_count]
    
    # Create a mask for bottom 50% std values
    bottom_50_percent_max_mask = torch.zeros_like(mean_vals, dtype=torch.bool)
    bottom_50_percent_max_mask[bottom_50_percent_max_indices] = True
    
    # Filter mean values
    std_vals_filtered = std_vals.clone()
    std_vals_filtered[bottom_50_percent_max_mask] = float('inf')
    
    std_vals = std_vals_filtered
    
    
    # Get indices of top 50% std values
    top_50_percent_std_count = int(0.5* len(std_vals))
    top_50_percent_std_indices = torch.argsort(std_vals)[:top_50_percent_std_count]
    
    #random indices
    random_indices = torch.randperm(len(std_vals))[:top_50_percent_std_count]
    
    # Create a mask for top 50% std values
    top_50_percent_std_mask = torch.zeros_like(std_vals, dtype=torch.bool)
    top_50_percent_std_mask[top_50_percent_std_indices] = True
    
    
    
    
    # Filter mean values
    mean_vals_filtered = mean_vals.clone()
    mean_vals_filtered[top_50_percent_std_mask] = float('-inf')
    
    # Compute mask
    return compute_max_mask(mean_vals_filtered, percent)

def compute_max_low_std_mask(mean_vals, std_vals, percent):
    # Get indices of bottom 50% std values
    bottom_50_percent_std_count = int(0.50 * len(std_vals))
    bottom_50_percent_std_indices = torch.argsort(std_vals)[:bottom_50_percent_std_count]
    
    # Create a mask for bottom 50% std values
    bottom_50_percent_std_mask = torch.zeros_like(std_vals, dtype=torch.bool)
    bottom_50_percent_std_mask[bottom_50_percent_std_indices] = True
    
    # Filter mean values
    mean_vals_filtered = mean_vals.clone()
    mean_vals_filtered[~bottom_50_percent_std_mask] = float('-inf')
    
    # Compute mask
    return compute_max_mask(mean_vals_filtered, percent)

def mask_range_distilbert(model, mask, fc_vals):
    mean = torch.tensor(np.mean(fc_vals, axis=0))
    std = torch.tensor(np.std(fc_vals, axis=0))
    mask = mask.to(torch.bool)
    a = 2.5
    lower_bound = torch.full_like(mean, torch.inf)
    lower_bound[~mask] = mean[~mask] - a*std[~mask]
    upper_bound = torch.full_like(mean, -torch.inf)
    upper_bound[~mask] = mean[~mask] + a*std[~mask]
    
    model.distilbert.transformer.mask_layer.lower_bound = lower_bound.to(device)
    model.distilbert.transformer.mask_layer.upper_bound = upper_bound.to(device)
    
    return model

def mask_range_bert(model, mask, fc_vals):
    mean = torch.tensor(np.mean(fc_vals, axis=0))
    std = torch.tensor(np.std(fc_vals, axis=0))
    mask = mask.to(torch.bool)
    a = 2.5
    lower_bound = torch.full_like(mean, torch.inf)
    lower_bound[~mask] = mean[~mask] - a*std[~mask]
    upper_bound = torch.full_like(mean, -torch.inf)
    upper_bound[~mask] = mean[~mask] + a*std[~mask]
    
    model.bert.encoder.mask_layer.lower_bound = lower_bound.to(device)
    model.bert.encoder.mask_layer.upper_bound = upper_bound.to(device)
    
    return model


def compute_std_high_max_mask(mean_vals, std_vals, percent):
    # Get indices of bottom 50% std values
    bottom_50_percent_max_count = int(0.20 * len(mean_vals))
    bottom_50_percent_max_indices = torch.argsort(mean_vals)[:bottom_50_percent_max_count]
    
    # Create a mask for bottom 50% std values
    bottom_50_percent_max_mask = torch.zeros_like(mean_vals, dtype=torch.bool)
    bottom_50_percent_max_mask[bottom_50_percent_max_indices] = True
    
    # Filter mean values
    std_vals_filtered = std_vals.clone()
    std_vals_filtered[bottom_50_percent_max_mask] = float('inf')
    
    # Compute mask
    return compute_std_mask(std_vals_filtered, percent)


def mask_distillbert(model, mask):
    model.distilbert.transformer.masking_layer = mask.to(device)
    return model

def mask_bert(model, mask):
    model.bert.encoder.masking_layer = mask.to(device)
    return model

def pad(batch):
    '''Pads to the longest sample'''
    f = lambda x: [sample[x] for sample in batch]
    words = f(0)
    is_heads = f(2)
    tags = f(3)
    seqlens = f(-1)
    maxlen = np.array(seqlens).max()

    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: <pad>
    x = f(1, maxlen)
    y = f(-2, maxlen)


    f = torch.LongTensor

    return words, f(x), is_heads, tags, f(y), seqlens

def train(model, iterator, optimizer, criterion):
    model.train()
    for i, batch in enumerate(iterator):
        words, x, is_heads, tags, y, seqlens = batch
        _y = y # for monitoring
        optimizer.zero_grad()
        enc, logits, y, _, _ = model(x, y) # logits: (N, T, VOCAB), y: (N, T)

        logits = logits.view(-1, logits.shape[-1]) # (N*T, VOCAB)
        y = y.view(-1)  # (N*T,)

        loss = criterion(logits, y)
        loss.backward()

        optimizer.step()

        if i%10==0: # monitoring
            print("step: {}, loss: {}".format(i, loss.item()))


def eval(model, iterator, idx2tag, tag2idx, tok):
    model.eval()
    Words, Is_heads, Tags, Y, Y_hat, Confidences = [], [], [], [], [], []
    encodings_by_tag = {}  # Dictionary to store encodings by tag
    tag_correct = {tag: 0 for tag in tag2idx}
    tag_total = {tag: 0 for tag in tag2idx}
    tag_confidence_sum = {tag: 0.0 for tag in tag2idx}

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_heads, tags, y, seqlens = batch
            enc, _, _, y_hat, confidence = model(x, y)  # y_hat: (N, T)
            Words.extend(words)
            Is_heads.extend(is_heads)
            Tags.extend(tags)
            Y.extend(y.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())
            Confidences.extend(confidence.cpu().numpy().tolist())

            for tag, encoding, h in zip(y[0].cpu().numpy(), enc[0], is_heads[0]):
                if h:
                    if tag not in encodings_by_tag:
                        encodings_by_tag[tag] = []
                    encodings_by_tag[tag].append(encoding.cpu().numpy())

    # Write results to file
    with open("result", 'w') as fout:
        for words, is_heads, tags, y_hat, confidences in zip(Words, Is_heads, Tags, Y_hat, Confidences):
            y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
            confs = [conf for head, conf in zip(is_heads, confidences) if head == 1]
            preds = [idx2tag[hat] for hat in y_hat]
            assert len(preds) == len(words.split()) == len(tags.split()) == len(confs)
            for w, t, p, c in zip(words.split()[1:-1], tags.split()[1:-1], preds[1:-1], confs[1:-1]):
                fout.write(f"{w} {t} {p} {c:.4f}\n")
                tag_total[t] += 1
                tag_confidence_sum[t] += c
                if t == p:
                    tag_correct[t] += 1
            fout.write("\n")

    # Calculate overall accuracy and confidence
    y_true = np.array([tag2idx[line.split()[1]] for line in open('result', 'r').read().splitlines() if len(line) > 0])
    y_pred = np.array([tag2idx[line.split()[2]] for line in open('result', 'r').read().splitlines() if len(line) > 0])
    confidences = np.array([float(line.split()[3]) for line in open('result', 'r').read().splitlines() if len(line) > 0])
    overall_acc = (y_true == y_pred).astype(np.int32).sum() / len(y_true)
    overall_confidence = confidences.mean()
    # print(f"Overall accuracy: {overall_acc:.3f}")
    # print(f"Overall confidence: {overall_confidence:.3f}")

    # Calculate and print per-token accuracies and confidences
    # print("\nPer-token accuracies and confidences:")
    for tag in tag2idx:
        if tag_total[tag] > 0:
            acc = tag_correct[tag] / tag_total[tag]
            conf = tag_confidence_sum[tag] / tag_total[tag]
            # print(f"{tag}: Acc: {acc:.3f} ({tag_correct[tag]}/{tag_total[tag]}), Conf: {conf:.3f}")
        else:
            print(f"{tag}: N/A (0 occurrences)")

    def get_specific_token_accuracy(token_id):
        if token_id not in idx2tag:
            return None, None, None, None
        token = idx2tag[token_id]
        if tag_total[token] == 0:
            return 0, 0, 0, 0
        token_acc = tag_correct[token] / tag_total[token]
        token_conf = tag_confidence_sum[token] / tag_total[token]
        other_correct = sum(tag_correct.values()) - tag_correct[token]
        other_total = sum(tag_total.values()) - tag_total[token]
        other_acc = other_correct / other_total if other_total > 0 else 0
        other_conf = (sum(tag_confidence_sum.values()) - tag_confidence_sum[token]) / other_total if other_total > 0 else 0
        return round(token_acc, 4), round(other_acc, 4), round(token_conf, 4), round(other_conf, 4)

    token_acc, other_acc, token_conf, other_conf = get_specific_token_accuracy(tok)
    # print(f"Specific token accuracy: {token_acc}, Specific token confidence: {token_conf}")
    # print(f"Other tokens accuracy: {other_acc}, Other tokens confidence: {other_conf}")

    return encodings_by_tag,(token_acc, token_conf), (other_acc, other_conf)