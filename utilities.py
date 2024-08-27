import torch
from tqdm import tqdm
import numpy as np
from models.distilbert import DistilBertForSequenceClassification
from transformers import AutoModelForSequenceClassification
from models.bert import BertForSequenceClassification
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def compute_accuracy(dataset, model, tokenizer, text_tag):
    correct = 0
    total_confidence = 0.0
    progress_bar = tqdm(total=len(dataset))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        for i in range(len(dataset)):
            text = dataset[i][text_tag]
            inputs = tokenizer(text, return_tensors="pt", max_length = 512).to(device)
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs[0], dim=-1)
            predicted_class_idx = torch.argmax(predictions).item()
            if predicted_class_idx == dataset[i]['label']:
                correct += 1
                total_confidence += predictions[0][predicted_class_idx].item()
            progress_bar.update(1)
    progress_bar.close()
    
    accuracy = correct / len(dataset)
    average_confidence = total_confidence / correct if correct > 0 else 0.0
    
    return round(accuracy,4), round(average_confidence,4)
def get_model_distilbert(directory_path, layer):
    base_path = os.path.join("model_weights", directory_path)
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    weights_path = os.path.join(base_path, "weights.pth")

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
    mask_intersection = mask_max * mask_std
    
    return mask_max, mask_std, mask_intersection, mask_max_low_std, mask_std_high_max

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


def compute_std_high_max_mask(mean_vals, std_vals, percent):
    # Get indices of bottom 50% std values
    bottom_50_percent_max_count = int(0.50 * len(mean_vals))
    bottom_50_percent_max_indices = torch.argsort(mean_vals, descending=True)[:bottom_50_percent_max_count]
    
    # Create a mask for bottom 50% std values
    bottom_50_percent_max_mask = torch.zeros_like(mean_vals, dtype=torch.bool)
    bottom_50_percent_max_mask[bottom_50_percent_max_indices] = True
    
    # Filter mean values
    std_vals_filtered = std_vals.clone()
    std_vals_filtered[~bottom_50_percent_max_mask] = float('inf')
    
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
    print(f"Overall accuracy: {overall_acc:.3f}")
    print(f"Overall confidence: {overall_confidence:.3f}")

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
    print(f"Specific token accuracy: {token_acc}, Specific token confidence: {token_conf}")
    print(f"Other tokens accuracy: {other_acc}, Other tokens confidence: {other_conf}")

    return encodings_by_tag