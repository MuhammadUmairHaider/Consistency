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
            total_confidence += predictions.gather(1, labels.unsqueeze(1)).squeeze(1).sum().item()
            
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
        average_confidence = total_confidence / total_samples
    
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

def compute_mask_probe(weights, percent):
    sorted_indices = torch.argsort(weights, descending=True)
    mask_count = int(percent * len(weights))
    mask = torch.ones_like(weights)
    mask[sorted_indices[:mask_count]] = 0.0
    return mask



def compute_avg_std(fc_vals, mask):
    fc_vals_array = np.array(fc_vals)
    
    normalized_vals = (fc_vals_array - fc_vals_array.min(axis=0)) / (fc_vals_array.max(axis=0) - fc_vals_array.min(axis=0))
    
    std_vals = np.std(normalized_vals, axis=0)
    
    return np.mean(std_vals[mask.bool()])



def compute_masks(fc_vals, percent):
    # Convert input to numpy array
    fc_vals_array = np.array(fc_vals)
    
    normalized_vals = (fc_vals_array - fc_vals_array.min(axis=0)) / (fc_vals_array.max(axis=0) - fc_vals_array.min(axis=0))
    
    # Compute statistics
    mean_vals = np.mean(np.abs(fc_vals_array), axis=0)
    std_vals = np.std(fc_vals_array, axis=0)
    min_vals = np.min(fc_vals_array, axis=0)
    max_vals = np.max(fc_vals_array, axis=0)
    
    # Normalize standard deviation
    std_vals_normalized = (std_vals - min_vals) / (max_vals - min_vals)
    
    # std_vals_normalized = std_vals
    
    # Convert to PyTorch tensors
    mean_vals_tensor = torch.from_numpy(mean_vals)
    std_vals_tensor = torch.from_numpy(std_vals_normalized)
    
    # Compute masks
    mask_max = compute_max_mask(mean_vals_tensor, percent)
    mask_std = compute_std_mask(std_vals_tensor, percent)
    
    mask_max_for_intersection = compute_max_mask(mean_vals_tensor, 0.4)
    mask_std_for_intersection = compute_std_mask(std_vals_tensor, 0.4)
    
    mask_intersection = 0#compute_intersection_mask(mask_max_for_intersection, mask_std_for_intersection, percent)
    
    mask_max_low_std = compute_max_low_std_mask(mean_vals_tensor, std_vals_tensor, percent)
    mask_std_high_max = compute_std_high_max_mask(mean_vals_tensor, std_vals_tensor, percent)
    mask_max_high_std = comute_max_high_std_mask(mean_vals_tensor, std_vals_tensor, percent)
    # mask_intersection = compute_intersection_mask(mask_max, mask_std, percent)
    mask_max_random_off = compute_max_random_off(mean_vals_tensor, percent)
    
    mask_random = compute_mask_random_off(mean_vals_tensor, percent)
    
    return mask_max, mask_std, mask_intersection, mask_max_low_std, mask_max_high_std, mask_std_high_max,mask_max_random_off, mask_random


def compute_intersection_mask(mask1: torch.Tensor, mask2: torch.Tensor, percent: float) -> torch.Tensor:
    """
    Compute intersection mask where values are masked (0) only when masked in both input masks.
    If intersection has more masked values than target percentage, randomly select percent values.
    Otherwise, mask additional values to reach the target percentage.
    
    Args:
        mask1 (torch.Tensor): First mask (1s for unmasked, 0s for masked positions)
        mask2 (torch.Tensor): Second mask (1s for unmasked, 0s for masked positions)
        percent (float): Desired percentage of total values to be masked (0 to 100)
    
    Returns:
        torch.Tensor: New mask with exactly percent values masked
    """
    # Start with all 1s
    result_mask = torch.ones_like(mask1)
    total_elements = result_mask.numel()
    target_masked = int((percent / 100) * total_elements)
    
    # Find intersection of masked positions (where both masks have 0s)
    intersection = (mask1 == 0) & (mask2 == 0)
    intersection_count = torch.sum(intersection).item()
    
    # Case 1: Intersection has more masked values than target
    if intersection_count > target_masked:
        # Get all intersection indices
        intersection_indices = torch.nonzero(intersection).squeeze()
        # Randomly select target_masked indices from intersection
        selected_indices = intersection_indices[torch.randperm(intersection_count)[:target_masked]]
        # Create new mask with only selected positions masked
        result_mask = torch.ones_like(mask1)
        result_mask.view(-1)[selected_indices] = 0
        
    # Case 2: Need to add more masked values
    else:
        # Start with intersection
        result_mask[intersection] = 0
        # Calculate how many additional values needed
        additional_needed = target_masked - intersection_count
        
        if additional_needed > 0:
            # Get indices of remaining 1s
            available_indices = torch.nonzero(result_mask == 1).squeeze()
            # Randomly select indices to mask
            indices_to_mask = available_indices[torch.randperm(len(available_indices))[:additional_needed]]
            result_mask.view(-1)[indices_to_mask] = 0
    
    return result_mask

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

def compute_intersection_mask(mask1, mask2, percent):
    intersection_mask = torch.logical_and(mask1, mask2)
    intersection_count = torch.count_nonzero(intersection_mask)
    mask_count = int(percent * len(intersection_mask))
    if intersection_count < mask_count:
        raise ValueError("Intersection mask has fewer elements than required")
    
    # Get indices of intersection mask
    intersection_indices = torch.where(intersection_mask)[0]
    
    # Create a mask for intersection values
    intersection_mask = torch.zeros_like(intersection_mask, dtype=torch.bool)
    intersection_mask[intersection_indices[:mask_count]] = True
    
    return intersection_mask
    

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

def compute_mask_random_off(mean_vals, percent):

    # Get indices of bottom 20% values
    bottom_20_percent_count = int(0.20 * len(mean_vals))
    bottom_20_percent_indices = torch.argsort(mean_vals)[:bottom_20_percent_count]
    
    # Create set of indices excluding bottom 20%
    all_indices = set(range(len(mean_vals)))
    excluded_indices = set(bottom_20_percent_indices.tolist())
    available_indices = list(all_indices - excluded_indices)
    
    # Randomly select from remaining indices
    num_to_select = int(percent * len(mean_vals))
    num_to_select = min(num_to_select, len(available_indices))  # Ensure we don't select more than available
    
    if num_to_select > 0:
        selected_positions = torch.randperm(len(available_indices))[:num_to_select]
        selected_indices = [available_indices[i] for i in selected_positions]
    else:
        selected_indices = []
    
    # Create final mask
    mask = torch.ones_like(mean_vals, dtype=torch.bool)
    mask[selected_indices] = 0.0
    
    return mask

def compute_max_random_off(mean_vals, percent):
    # Get indices of bottom 50% std values
    bottom_50_percent_std_count = int(0.50 * len(mean_vals))
    #pick random indices
    random_indices = torch.randperm(len(mean_vals))[:bottom_50_percent_std_count]
    # bottom_50_percent_std_indices = torch.argsort(std_vals)[:bottom_50_percent_std_count]
    
    # Create a mask for bottom 50% std values
    bottom_50_percent_std_mask = torch.zeros_like(mean_vals, dtype=torch.bool)
    bottom_50_percent_std_mask[random_indices] = True
    
    # Filter mean values
    mean_vals_filtered = mean_vals.clone()
    mean_vals_filtered[~bottom_50_percent_std_mask] = float('-inf')
    
    # Compute mask
    return compute_max_mask(mean_vals_filtered, percent)

def mask_range_distilbert(tao,model, mask, fc_vals, fc_vals2):
    
    mean = torch.tensor(np.mean(fc_vals, axis=0))
    mean2 = torch.tensor(np.mean(fc_vals2, axis=0))
    std = torch.tensor(np.std(fc_vals, axis=0))
    mask = mask.to(torch.bool)
    lower_bound = torch.full_like(mean, torch.inf)
    lower_bound[~mask] = mean[~mask] - tao*std[~mask]
    upper_bound = torch.full_like(mean, -torch.inf)
    upper_bound[~mask] = mean[~mask] + tao*std[~mask]
    
    model.distilbert.transformer.mask_layer.lower_bound = lower_bound.to(device)
    model.distilbert.transformer.mask_layer.upper_bound = upper_bound.to(device)
    model.distilbert.transformer.mask_layer.replacement_values = mean2.to(device)
    
    return model

def mask_range_bert(tao, model, mask, fc_vals):
    mean = torch.tensor(np.mean(fc_vals, axis=0))
    std = torch.tensor(np.std(fc_vals, axis=0))
    mask = mask.to(torch.bool)
    lower_bound = torch.full_like(mean, torch.inf)
    lower_bound[~mask] = mean[~mask] - tao*std[~mask]
    upper_bound = torch.full_like(mean, -torch.inf)
    upper_bound[~mask] = mean[~mask] + tao*std[~mask]
    
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



#gpt utils
import nethook
def manual_generate(model, tokenizer, input_ids, attention_mask, max_length):
    device = input_ids.device
    batch_size = input_ids.shape[0]
    
    # Initialize the output tensor with the input_ids
    generated = input_ids.clone()
    
    confidences = torch.tensor([])
    all_fc_vals = torch.tensor([])
    
    with torch.no_grad():
        for _ in range(max_length - input_ids.shape[1]):
            # Forward pass
            with nethook.TraceDict(model, ['transformer.mask_layer']) as ret:
                outputs = model(input_ids=generated, attention_mask=attention_mask)
                
                fc1_vals = [
                    ret[layer_fc1_vals].output[:,-1,:].to('cpu')
                    for layer_fc1_vals in ret
                ]
                all_fc_vals = torch.cat([all_fc_vals, torch.stack(fc1_vals, dim=1).unsqueeze(1)], dim=1)
                
                next_token_logits = outputs.logits[:, -1, :]
                
                # Apply greedy decoding (argmax)
                next_tokens = torch.argmax(next_token_logits, dim=-1)
                
                # Append the confidence of the predicted token
                confidences = torch.cat([confidences, torch.nn.functional.softmax(next_token_logits, dim=-1).unsqueeze(1).to('cpu')], dim=1)
                
                # Append the new tokens
                generated = torch.cat([generated, next_tokens.unsqueeze(-1)], dim=-1)
                
                # Update attention mask
                attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1), dtype=torch.long, device=device)], dim=1)
    
    return generated, confidences, all_fc_vals



import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
def collate_fn(batch):
    return {
        'input_ids': torch.stack([torch.tensor(item['input_ids']) for item in batch]),
        'attention_mask': torch.stack([torch.tensor(item['attention_mask']) for item in batch]),
        'label': torch.stack([torch.tensor(item['label']) for item in batch])
    }



def manual_generate_v2(model, input_ids, attention_mask):
    
    with torch.no_grad():
        with nethook.TraceDict(model, ['transformer.mask_layer']) as ret:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            fc1_vals = [
                    ret[layer_fc1_vals].output[:,-1,:].to('cpu')
                    for layer_fc1_vals in ret
                ]
        next_token_logits = outputs.logits[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1)
        confidences = torch.nn.functional.softmax(next_token_logits, dim=-1)
        
        return next_tokens, confidences, fc1_vals[0]
        

def evaluate_gpt2_classification(lab ,model, eval_dataset, tokenizer, batch_size=32):
    
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    confidence = 0
    all_hidden = []
    correct = 0
    j = 0
    dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    for item in tqdm(dataloader, desc="Evaluating"):
       input_ids = torch.tensor(item['input_ids']).to(device)
       attention_mask = torch.tensor(item['attention_mask']).to(device)
        
       generated_token, confidences, fc_vals = manual_generate_v2(model,input_ids,attention_mask)
       
       for true_label, predicted_token, conf, fc in zip(item[lab], generated_token, confidences, fc_vals):
            true_label = tokenizer.encode(eval_dataset.features[lab].int2str(true_label.item()), add_special_tokens=False, truncation=True, return_tensors='pt').squeeze()
            
            predicted_label = predicted_token
            confidence += conf[true_label].cpu().numpy().item()
            j += 1
            if predicted_label == true_label:
                correct += 1
            all_hidden.append(fc.numpy())
    if j == 0:
        return 0, 0, []
            
    return round(correct/j,4), round(confidence/j,4), all_hidden
    


def evaluate_gpt2_classification_batch(model, eval_dataset, tokenizer, batch_size=32):
    
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.config.pad_token_id = tokenizer.pad_token_id 
    
    all_predictions = []
    all_labels = []
    confidence = 0
    j = 0 
    all_hidden = []
    dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    for item in tqdm(dataloader, desc="Evaluating"):
        input_ids = torch.tensor(item['input_ids']).to(device)
        attention_mask = torch.tensor(item['attention_mask']).to(device)
        
        generated_sequences, confidences, fc_vals = manual_generate(model,tokenizer,input_ids,attention_mask,400)
        
        

        label_token_ids = tokenizer.encode('[Label]', add_special_tokens=False)
        label_len = len(label_token_ids)

        for input_id, generated_sequence, conf, fc in zip(input_ids,generated_sequences, confidences, fc_vals):
            generated_sequence = generated_sequence[input_id.shape[0]:].to('cpu')
            label_positions = []
            for i in range(len(generated_sequence) - label_len + 1):
                if generated_sequence[i:i+label_len].tolist() == label_token_ids:
                    label_positions.append(i)
                    break
                
            full_text = tokenizer.decode(input_id)
            true_label = full_text.split("[Label] ")[1].split("<|endoftext|>")[0]
            
            true_label_tokenized = tokenizer.encode(true_label)

            if not label_positions or label_positions[0]+1 == len(generated_sequence):
                predicted_label = "No label found"
            else:
                for pos in label_positions:
                    predicted_label = tokenizer.decode(generated_sequence[pos+1])
                    
                    hidden_dim = fc[pos+1]
                    confidence += conf[pos+1][true_label_tokenized].cpu().numpy().item()
                    # if(predicted_label == true_label):
                        
                    j += 1
                    all_hidden.append(hidden_dim.numpy())

            all_predictions.append(predicted_label)
            all_labels.append(true_label)
    
    if not all_labels or not all_predictions:
        print("No labels were extracted. Check if '[Label]' token exists in the tokenized text.")
        return 0, "No labels extracted", [], []

    accuracy = accuracy_score(all_labels, all_predictions)
    
    unique_labels = list(set(all_labels + all_predictions))
    
    try:
        report = classification_report(all_labels, all_predictions, labels=unique_labels, target_names=unique_labels)
    except ValueError as e:
        report = f"Unable to generate classification report: {str(e)}"
    
    confidence = confidence/j if j > 0 else 0
    
    return round(accuracy,4), round(confidence,4), all_hidden, report, all_labels, all_predictions 

def mask_range_gpt(model, mask, fc_vals, tao):
    mean = torch.tensor(np.mean(fc_vals, axis=0))
    std = torch.tensor(np.std(fc_vals, axis=0))
    mask = mask.to(torch.bool)
    
    lower_bound = torch.full_like(mean, torch.inf)
    lower_bound[~mask] = mean[~mask] - tao*std[~mask]
    upper_bound = torch.full_like(mean, -torch.inf)
    upper_bound[~mask] = mean[~mask] + tao*std[~mask]
    
    model.transformer.mask_layer.lower_bound = lower_bound.to(device)
    model.transformer.mask_layer.upper_bound = upper_bound.to(device)
    
    return model

def reset_gpt(model):
    model.transformer.mask_layer.lower_bound = torch.tensor(float('inf')).to(device)
    model.transformer.mask_layer.upper_bound = torch.tensor(float('-inf')).to(device)
    return model

#Lama Utilities

import nethook
def manual_generate_llma(model, input_ids, attention_mask, max_length):
    device = input_ids.device
    batch_size = input_ids.shape[0]
    
    # Initialize the output tensor with the input_ids
    generated = input_ids.clone()
    
    confidences = torch.tensor([])
    all_fc_vals = torch.tensor([])
    fc = torch.tensor([])
    with torch.no_grad():
        for _ in range(max_length - input_ids.shape[1]):
            # Forward pass
                with nethook.TraceDict(model, ['model.mask_layer']) as ret:
                    outputs = model(input_ids=generated, attention_mask=attention_mask)
                    fc1_vals = [
                    ret[layer_fc1_vals].output[:,-1,:].to('cpu')
                    for layer_fc1_vals in ret
                ]
                all_fc_vals = torch.cat([all_fc_vals, torch.stack(fc1_vals, dim=1).unsqueeze(1)], dim=1)
                next_token_logits = outputs.logits[:, -1, :]
                
                # Apply greedy decoding (argmax)
                next_tokens = torch.argmax(next_token_logits, dim=-1)
                
                # Append the confidence of the predicted token
                confidences = torch.cat([confidences, torch.nn.functional.softmax(next_token_logits, dim=-1).unsqueeze(1).to('cpu')], dim=1)
                
                # Append the new tokens
                generated = torch.cat([generated, next_tokens.unsqueeze(-1)], dim=-1)
                
                # Update attention mask
                attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1), dtype=torch.long, device=device)], dim=1)
    
    return generated, confidences, all_fc_vals


from tqdm.auto import tqdm
import torch
# def evaluate_llma_classification(model, eval_dataset, tokenizer):

#     # Create label mapping
#     label_mapping = {
#         0: "sadness",
#         1: "joy",
#         2: "love",
#         3: "anger",
#         4: "fear",
#         5: "surprise"
#     }


#     # Configure progress bar for the combined dataset
#     progress_bar = tqdm(eval_dataset, desc="Processing examples")

#     i = 0
#     correct = 0
#     confidence = 0
#     fc_vals = []
#     model.to('cuda')
#     model.eval()

#     for example in progress_bar:
#         prompt = '''Choose from one of these: anger, fear, joy, love, sadness, surprise
#     {{"I can't believe how wonderful this day has been!":joy}}
#     {{"Missing you more with each passing day":sadness}}
#     {{"How dare they treat me like this!":anger}}
#     {{"I'm getting butterflies just thinking about tomorrow":fear}}
#     {{"You mean everything to me": ["love"]}}
#     {{"I didn't expect this to happen at all":surprise}}
#     {{"{}":'''.format(example['text'])
        
#         input_ids = tokenizer([prompt, prompt], return_tensors='pt')
        
#         # Generate output
#         output = manual_generate_llma(
#             model, 
#             input_ids['input_ids'].to('cuda'), 
#             input_ids['attention_mask'].to('cuda'), 
#             input_ids['input_ids'].shape[1]+5
#         )
        
#             # Get predicted label
#         predicted_label = tokenizer.decode(
#             output[0][0][input_ids['input_ids'].shape[1]:]
#         ).split('}')[0]
        
#         # Get true label text using the mapping
#         true_label_text = label_mapping[example['label']]
        
#         i += 1
#         if true_label_text == predicted_label:
#             correct += 1
            
#         # else:
#         #     print(f"Text: {example['text']}, True label: {true_label_text}, Predicted label: {predicted_label}")
        
#         label_tok = tokenizer.encode(true_label_text)[1]
        
#         # Update progress bar description with current accuracy
#         progress_bar.set_description(f"Accuracy: {round(correct/i,3)*100}%")
#         confidence += output[1][0][0][label_tok].item()
#         fc_vals.append(output[2][0][0].squeeze())
        
        

#     return round(correct/i,4), round(confidence/i,4), fc_vals



def evaluate_llma_classification(model, eval_dataset, tokenizer):

    # Create label mapping
    label_mapping = label_mapping = {0: 'negative', 1: 'positive'}



    # Configure progress bar for the combined dataset
    progress_bar = tqdm(eval_dataset, desc="Processing examples")

    i = 0
    correct = 0
    confidence = 0
    fc_vals = []
    model.to('cuda')
    model.eval()

    for example in progress_bar:
        prompt = '''Choose from one of these sentiments: negative, positive. These are your only choises. Be careful distinguishing between similar sentiments. These are real reviews.
{{badly-rendered cgi effects :negative}}
{{it feels more like the pilot episode of a tv series than a feature film :negative}}
{{if you liked the previous movies in the series :positive}}
{{required to balance all the formulaic equations in the long-winded heist comedy who is cletis tout ? :negative}}
{{a load of clams left in the broiling sun for a good three days :negative}}
{{if you liked the previous movies in the series , you 'll have a good time with this one too :positive}}
{{it 's a long way from orwell 's dark , intelligent warning cry ( 1984 ) to the empty stud knockabout of equilibrium , and what once was conviction is now affectation :positive}}
{{a smoother , more focused :positive}}
{{the genuinely funny jokes are few and far between :negative}}
{{have stayed there :positive}}
{{"{}":"'''.format(example['text'])
        
        input_ids = tokenizer([prompt, prompt], return_tensors='pt')
        
        # Generate output
        output = manual_generate_llma(
            model, 
            input_ids['input_ids'].to('cuda'), 
            input_ids['attention_mask'].to('cuda'), 
            input_ids['input_ids'].shape[1]+8
        )
        
            # Get predicted label
        predicted_label = tokenizer.decode(
            output[0][0][input_ids['input_ids'].shape[1]:]
        ).split('}')[0]
        
        predicted_label = predicted_label.strip('"')
        
        # Get true label text using the mapping
        true_label_text = label_mapping[example['label']]
        
        i += 1
        if true_label_text == predicted_label:
            correct += 1
            
        # else:
        #     print(f"Text: {example['text']}, True label: {true_label_text}, Predicted label: {predicted_label}")
        
        label_tok = tokenizer.encode(true_label_text)[1]
        
        # Update progress bar description with current accuracy
        progress_bar.set_description(f"Accuracy: {round(correct/i,3)*100}%")
        confidence += output[1][0][0][label_tok].item()
        fc_vals.append(output[2][0][0].squeeze())
        
        

    return round(correct/i,4), round(confidence/i,4), fc_vals

def reset_llma(model):
    model.model.mask_layer.lower_bound = torch.tensor(float('inf')).to(device)
    model.model.mask_layer.upper_bound = torch.tensor(float('-inf')).to(device)
    return model


def mask_range_llma(model, mask, fc_vals, tao):
    mean = torch.tensor(np.mean(fc_vals, axis=0))
    std = torch.tensor(np.std(fc_vals, axis=0))
    mask = mask.to(torch.bool)
    
    lower_bound = torch.full_like(mean, torch.inf)
    lower_bound[~mask] = mean[~mask] - tao*std[~mask]
    upper_bound = torch.full_like(mean, -torch.inf)
    upper_bound[~mask] = mean[~mask] + tao*std[~mask]
    
    model.model.mask_layer.lower_bound = lower_bound.to(device)
    model.model.mask_layer.upper_bound = upper_bound.to(device)
    
    return model