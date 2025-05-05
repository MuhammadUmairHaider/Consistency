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
    sorted_indices = torch.argsort(torch.abs(weights), descending=True)
    mask_count = int(percent * len(weights))
    mask = torch.ones_like(weights)
    mask[sorted_indices[:mask_count]] = 0.0
    return mask



def compute_avg_mad(fc_vals, mask):
    fc_vals_array = np.array(fc_vals)
    
    # normalized_vals = (fc_vals_array - fc_vals_array.min(axis=0)) / (fc_vals_array.max(axis=0) - fc_vals_array.min(axis=0))
    
    std_vals = mad(fc_vals_array)
    
    return np.mean(std_vals[mask.bool()])


def mad(data):
    median = np.median(data, axis=0)
    deviations = np.abs(data - median)
    mad_value = np.median(deviations, axis=0)
    return mad_value

def z_score(data):
    
    #4200,768
    """
    Calculate z-scores along axis 0 (columns) of a 2D array.
    
    Parameters:
    data : numpy.ndarray
        2D input array where rows are observations and columns are features
        
    Returns:
    numpy.ndarray
        Array of same shape as input with z-scores calculated for each column
    """
    # Calculate mean along axis 0
    mean = np.mean(data, axis=0)
    
    #768
    
    # Calculate standard deviation along axis 0
    std = np.std(data, axis=0)  # ddof=1 for sample standard deviation
    
    # Avoid division by zero
    # std = np.where(std == 0, 1, std)
    
    # Calculate z-scores
    
    # 4200,768
    z_scores = (data - mean) / std
    z_scores = np.mean(np.abs(z_scores), axis=0)
    
    return z_scores

def compute_pca(vectors, n_components=768):
    """
    Compute PCA for a list of vectors using PyTorch SVD on GPU.
    
    Parameters:
    - vectors: List of vectors or tensor of shape (n_vectors, 768)
    - n_components: Number of principal components to keep
    
    Returns:
    - pca_components: Principal components, shape (n_components, 768)
    - explained_variance_ratio: Explained variance ratio for each component
    - transformed_data: Data transformed into PCA space, shape (n_vectors, n_components)
    - feature_ranking: Indices of features sorted by importance
    - feature_importance: Importance score for each feature
    """
    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert to tensor if not already
    if not isinstance(vectors, torch.Tensor):
        data = torch.tensor(vectors, dtype=torch.float32)
    else:
        data = vectors.clone().to(dtype=torch.float32)
    
    # Move to GPU
    data = data.to(device)
    
    # Center the data (subtract mean)
    mean = torch.mean(data, dim=0, keepdim=True)
    centered_data = data - mean
    
    # Scale by sqrt(n-1) for numerical stability
    n_samples = centered_data.shape[0]
    scaled_data = centered_data / torch.sqrt(torch.tensor(n_samples - 1, device=device))
    
    # Compute SVD
    U, S, V = torch.linalg.svd(scaled_data, full_matrices=False)
    
    # V is already transposed in PyTorch's SVD implementation
    components = V[:n_components]
    
    # Eigenvalues are squares of singular values
    eigenvalues = S[:n_components]**2
    
    # Calculate explained variance ratio
    total_variance = torch.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_variance
    
    # Project data onto principal components
    transformed_data = torch.matmul(centered_data, components.T)
    
    # Compute feature importance
    feature_importance = torch.sum(torch.abs(components), dim=0)
    
    # Get feature ranking (descending order of importance)
    feature_ranking = torch.argsort(feature_importance, descending=True)
    
    return {
        'pca_components': components,
        'explained_variance_ratio': explained_variance_ratio,
        'transformed_data': transformed_data,
        'feature_ranking': feature_ranking,
        'feature_importance': feature_importance
    }

from scipy import stats

def compute_kst_mask(fc_vals, percent):
    
    n_neurons = fc_vals.shape[1]
    
    kst_score = np.zeros(n_neurons)
    for neuron in range(n_neurons):
        neuron_data = fc_vals[:, neuron]
        # Normalize data
        normalized_data = (neuron_data - np.mean(neuron_data)) / np.std(neuron_data)
        # Run KS test against standard normal
        ks_stat, _ = stats.kstest(normalized_data, 'norm')
        
        kst_score[neuron] = ks_stat
        
    kst_score = torch.tensor(kst_score)
    
    sorted_indices = torch.argsort(kst_score, descending=False)
    
    mask_count = int(percent * n_neurons)
    mask = torch.ones_like(kst_score)
    mask[sorted_indices[:mask_count]] = 0.0
    
    return mask

def compute_masks(fc_vals, percent):
    # Convert input to numpy array
    fc_vals_array = np.array(fc_vals)
    print("Shape of the FC:",fc_vals_array.shape)
    
    
    
    # normalized_vals = (fc_vals_array - fc_vals_array.min(axis=0)) / (fc_vals_array.max(axis=0) - fc_vals_array.min(axis=0))
    
    # Compute statistics
    mean_vals = np.mean(np.abs(fc_vals_array), axis=0)
    std_vals = np.std(fc_vals_array, axis=0)
    
    
    # 500, 768 -> 768
    # 
    min_vals = np.min(std_vals, axis=0)
    max_vals = np.max(std_vals, axis=0)
    
    # Normalize standard deviation
    std_vals_normalized = mad(fc_vals_array)#(std_vals - min_vals) / (max_vals - min_vals)
    
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
    # kst_mask = compute_kst_mask(fc_vals_array, percent)
    return mask_max, mask_std, mask_intersection, mask_max_low_std, mask_max_high_std, mask_std_high_max,mask_max_random_off, mask_random#, kst_mask


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
    print(values.shape)
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
    
    
    
    # Get indices of bottom 50% std values
    bottom_50_percent_std_count = int(0.50 * len(std_vals))
    bottom_50_percent_std_indices = torch.argsort(std_vals, descending=True)[:bottom_50_percent_std_count]
    
    # Create a mask for bottom 50% std values
    bottom_50_percent_std_mask = torch.zeros_like(std_vals, dtype=torch.bool)
    bottom_50_percent_std_mask[bottom_50_percent_std_indices] = True
    
    # Filter mean values
    mean_vals_filtered = mean_vals.clone()
    mean_vals_filtered[~bottom_50_percent_std_mask] = float('-inf')
    
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

def mask_range_distilbert(tao,model, mask, fc_vals):
    
    mean = torch.tensor(np.mean(fc_vals, axis=0))
    # mean2 = torch.tensor(np.mean(fc_vals2, axis=0))
    std = torch.tensor(np.std(fc_vals, axis=0))
    mask = mask.to(torch.bool)
    lower_bound = torch.full_like(mean, torch.inf)
    lower_bound[~mask] = mean[~mask] - tao*std[~mask]
    upper_bound = torch.full_like(mean, -torch.inf)
    upper_bound[~mask] = mean[~mask] + tao*std[~mask]
    
    model.distilbert.transformer.mask_layer.lower_bound = lower_bound.to(device)
    model.distilbert.transformer.mask_layer.upper_bound = upper_bound.to(device)
    # model.distilbert.transformer.mask_layer.replacement_values = mean2.to(device)
    
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

def mask_gpt2(model, mask):
    model.transformer.mask_m_layer = mask.to(device)
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
        fc1_vals = fc1_vals[0]
        
        return next_tokens, confidences, fc1_vals#.reshape(-1, fc1_vals.shape[2])
        

def evaluate_gpt2_classification(lab ,model, eval_dataset, tokenizer, batch_size=128):
    
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    confidence = 0
    all_hidden = []
    correct = 0
    j = 0
    return_dataset = []
    dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    for item in tqdm(dataloader, desc="Evaluating"):
       input_ids = torch.tensor(item['input_ids']).to(device)
       attention_mask = torch.tensor(item['attention_mask']).to(device)
        
       with torch.no_grad():
           generated_token, confidences, fc_vals = manual_generate_v2(model,input_ids,attention_mask)
       
       for true_label, predicted_token, conf, fc in zip(item[lab], generated_token, confidences, fc_vals):
            true_label = tokenizer.encode(eval_dataset.features[lab].int2str(true_label.item()), add_special_tokens=False, truncation=True, return_tensors='pt').squeeze()
            
            predicted_label = predicted_token
            confidence += conf[true_label].cpu().numpy().item()
            j += 1
            if predicted_label == true_label:
                correct += 1
                return_dataset.append(item)
            all_hidden.append(fc.numpy())
    if j == 0:
        return 0, 0, []
            
    return round(correct/j,4), round(confidence/j,4), all_hidden, return_dataset
    


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

def mask_range_gpt(model, mask, fc_vals, tao, all_fc_vals):
    
    # all_fc_vals = np.concatenate(all_fc_vals)
    
    # all_fc_vals = torch.cat(all_fc_vals, dim=1)

    # mean_comp = torch.tensor(np.mean(all_fc_vals, axis=0))
    
    
    mean = torch.tensor(np.mean(fc_vals, axis=0))
    std = torch.tensor(np.std(fc_vals, axis=0))
    mask = mask.to(torch.bool)
    
    lower_bound = torch.full_like(mean, torch.inf)
    lower_bound[~mask] = mean[~mask] - tao*std[~mask]
    upper_bound = torch.full_like(mean, -torch.inf)
    upper_bound[~mask] = mean[~mask] + tao*std[~mask]
    
    model.transformer.mask_layer.lower_bound = lower_bound.to(device)
    model.transformer.mask_layer.upper_bound = upper_bound.to(device)
    # model.transformer.mask_layer.replacement_values = mean_comp.to(device)
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


#SST2
# def evaluate_llma_classification(model, eval_dataset, tokenizer):

#     # Create label mapping
#     label_mapping = label_mapping = {0: 'negative', 1: 'positive'}



#     # Configure progress bar for the combined dataset
#     progress_bar = tqdm(eval_dataset, desc="Processing examples")

#     i = 0
#     correct = 0
#     confidence = 0
#     fc_vals = []
#     model.to('cuda')
#     model.eval()

#     for example in progress_bar:
#         prompt = '''Choose from one of these sentiments: negative, positive. These are your only choises. Be careful distinguishing between similar sentiments. These are real reviews.
# {{badly-rendered cgi effects :negative}}
# {{it feels more like the pilot episode of a tv series than a feature film :negative}}
# {{if you liked the previous movies in the series :positive}}
# {{required to balance all the formulaic equations in the long-winded heist comedy who is cletis tout ? :negative}}
# {{a load of clams left in the broiling sun for a good three days :negative}}
# {{if you liked the previous movies in the series , you 'll have a good time with this one too :positive}}
# {{it 's a long way from orwell 's dark , intelligent warning cry ( 1984 ) to the empty stud knockabout of equilibrium , and what once was conviction is now affectation :positive}}
# {{a smoother , more focused :positive}}
# {{the genuinely funny jokes are few and far between :negative}}
# {{have stayed there :positive}}
# {{"{}":"'''.format(example['text'])
        
#         input_ids = tokenizer([prompt, prompt], return_tensors='pt')
        
#         # Generate output
#         output = manual_generate_llma(
#             model, 
#             input_ids['input_ids'].to('cuda'), 
#             input_ids['attention_mask'].to('cuda'), 
#             input_ids['input_ids'].shape[1]+8
#         )
        
#             # Get predicted label
#         predicted_label = tokenizer.decode(
#             output[0][0][input_ids['input_ids'].shape[1]:]
#         ).split('}')[0]
        
#         predicted_label = predicted_label.strip('"')
        
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


#IMDB

# def evaluate_llma_classification(model, eval_dataset, tokenizer):

#     # Create label mapping
#     label_mapping = label_mapping = {0: 'neg', 1: 'pos'}



#     # Configure progress bar for the combined dataset
#     progress_bar = tqdm(eval_dataset, desc="Processing examples")

#     i = 0
#     correct = 0
#     confidence = 0
#     fc_vals = []
#     model.to('cuda')
#     model.eval()

#     for example in progress_bar:
#         prompt = '''Choose from one of these categories: neg, pos. Be careful distinguishing between similar categories.
# {{The only reason I DVRd this movie was because 1. I live in Cleveland and Shaq plays basketball for us now and 2. I've always heard how awful it was. The movie did not disappoint. The best parts were Shaq's outfits. The worst parts were, well, just about everything else. My 12 year old son and I just squirmed and couldn't look at the screen when Shaq started rapping and we kept wondering why Max didn't wish for Kazzam to fix that front tooth of his! But for all it's terribleness we just couldn't stop watching it, the story sucked you in, like a black hole or quicksand or a tar pit, it was hypnotic. But it was worth it for the laughs and just to say that we actually watched "Kazzam".:neg}}
# {{Timberlake's performance almost made attack the screen. It wasn't all bad, I just think the reporters role was wrong for him.<br /><br />LL Cool J played the typical rapper role, toughest,baddest guy around. I don't think the cracked a smile in the whole movie, not even when proposed to his girlfriend.<br /><br />Morgan Freeman pretty much carried the whole movie. He was has some funny scenes which are the high point of the movie.<br /><br />Kevin Spacey wasn't good or bad he was just "there".<br /><br />Overall it's a Dull movie. bad plot. a lot of bad acting or wrong roles for actors.:neg}}
# {{'Deliverance' is a brilliant condensed epic of a group of thoroughly modern men who embark on a canoe trip to briefly commune with nature, and instead have to fight for their sanity, their lives, and perhaps even their souls. The film has aged well. Despite being made in the early Seventies, it certainly doesn't look particularly dated. It still possesses a visceral punch and iconic status as a dramatic post-'Death of the Sixties' philosophical-and-cultural shock vehicle. There are very few films with similar conceits that can compare favourably to it, although the legendary Sam Peckinpah's stuff would have to be up there. Yes, there has been considerable debate and discussion about the film's most confronting scene (which I won't expand upon here) - and undoubtedly one of the most confronting scenes in the entire history of the cinematic medium - but what surprises about this film is how achingly beautiful it is at times. This seems to be generally overlooked (yet in retrospect quite understandably so). The cinematography that captures the essence of the vanishing, fragile river wilderness is often absolutely stunning, and it counterbalances the film as, in a moment of brief madness, we the viewers - along with the characters themselves - are plunged into unrelenting nightmare. 'Deliverance's narrative is fittingly lean and sinewy, and it is surprising how quickly events unfold from point of establishment, through to crisis, and aftermath. It all takes place very quickly, which lends a sense of very real urgency to the film. The setting is established effectively through the opening credits. The characters are all well-drawn despite limited time spent on back story. We know just enough about them to know them for the kind of man they are, like them and ultimately fear for them when all goes to hell. The conflict and violence within the movie seems to erupt out of nowhere, with a frightening lack of logic. This is author James Dickey's theme - that any prevailing romanticism about the nature of Man's perceived inherent 'goodness' can only wilt and die when his barely suppressed animal instincts come to the fore. There are no demons or bogeymen here. The predatory hillbillies - as the film's central villains - are merely crude, terrifyingly amoral cousins of our protagonists. They shock because their evil is petty and tangible. The film has no peripheral characters. All reflect something about the weaknesses and uncertainties of urbanised Homo Sapiens in the latter 20th century, and all are very real and recognisable. Burt Reynolds is wonderful in this movie as the gung-ho and almost fatally over-confident Survivalist, Lewis, and it is a shame to think that he really couldn't recapture his brief moment of dramatic glory throughout the rest of his still sputtering up-and-down career ('Boogie Nights' excluded, perhaps). Trust me, if your are not a Reynolds fan, you WILL be impressed with his performance here. John Voight is his usual effortlessly accomplished self, and Ned Beatty and Ronny Cox both make significant contributions. This is simply a great quartet of actors. To conclude, I must speculate as to if and when 'Deliverance' author James Dickey's 'To the White Sea' will be made. For those that enjoyed (?) this film, TTWS is a similarly harrowing tale of an American Air Force pilot's struggle for survival after being shot down over the Japanese mainland during WW2. It's more of the typically bleak existentialism and primordial savagery that is Dickey's trademark, but it has all the makings of a truly spectacular, poetic cinematic experience. There was the suggestion a few years ago that the Coen brothers might be producing it, but that eventually came to nothing. Being an avid Coen-o-phile it disappoints me to think what might have been had they gotten the green light on TTWS, rather than their last couple of relatively undistinguished efforts. Returning to 'Deliverance', it's impossible to imagine a movie of such honest, unnerving brutality being made in these times, and that is pretty shameful. We, the cinema-going public, are all the poorer for this.:pos}}
# {{"{}":'''.format(example['text'])
        
#         input_ids = tokenizer([prompt, prompt], return_tensors='pt')
        
#         # Generate output
#         output = manual_generate_llma(
#             model, 
#             input_ids['input_ids'].to('cuda'), 
#             input_ids['attention_mask'].to('cuda'), 
#             input_ids['input_ids'].shape[1]+4
#         )
        
#             # Get predicted label
            
#         st = output[0][0][input_ids['input_ids'].shape[1]:]
#         predicted_label = tokenizer.decode(st).split('}')[0]
#         try:
#             tok_lab = tokenizer.encode(predicted_label)[1]
            
#         except:
#             print(st)
#             print(predicted_label)
#             # print(tokenizer.encode(predicted_label))
        
#         # Get predicted label position in st
#         label_positions = []
#         for j in range(len(st)):
#             if st[j] == tok_lab:
#                 label_positions.append(j)
#                 break
        
        
        
#         predicted_label = predicted_label.strip('"')
        
#         # Get true label text using the mapping
#         true_label_text = label_mapping[example['label']]
        
#         i += 1
#         if true_label_text == predicted_label:
#             correct += 1
#             fc_vals.append(output[2][0][label_positions[0]].squeeze())
            
#         # else:
#         #     print(f"Text: {example['text']}, True label: {true_label_text}, Predicted label: {predicted_label}")
        
#         label_tok = tokenizer.encode(true_label_text)[1]
        
#         # Update progress bar description with current accuracy
#         progress_bar.set_description(f"Accuracy: {round(correct/i,3)*100}%")
#         confidence += output[1][0][0][label_tok].item()
        
        
        

#     return round(correct/i,4), round(confidence/i,4), fc_vals


from tqdm.auto import tqdm
import torch
def evaluate_llma_classification(model, eval_dataset, tokenizer):

    # Create label mapping
    label_mapping = {0: 'Company', 1: 'EducationalInstitution', 2: 'Artist', 3: 'Athlete', 4: 'OfficeHolder', 5: 'MeanOfTransportation', 6: 'Building', 7: 'NaturalPlace', 8: 'Village', 9: 'Animal', 10: 'Plant', 11: 'Album', 12: 'Film', 13: 'WrittenWork'}



    # Configure progress bar for the combined dataset
    progress_bar = tqdm(eval_dataset, desc="Processing examples")

    i = 0
    correct = 0
    confidence = 0
    fc_vals = []
    model.to('cuda')
    model.eval()

    for example in progress_bar:
        prompt = '''Choose from one of these categories: Company, EducationalInstitution, Artist, Athlete, OfficeHolder, MeanOfTransportation, Building, NaturalPlace, Village, Animal, Plant, Album, Film, WrittenWork. Be careful distinguishing between similar categories.

{{ Abbott of Farnham E D Abbott Limited was a British coachbuilding business based in Farnham Surrey trading under that name from 1929. A major part of their output was under sub-contract to motor vehicle manufacturers. Their business closed in 1972.:Company}}

{{ Dubai Gem Private School (DGPS) is a British school located in the Oud Metha area of Dubai United Arab Emirates. Dubai Gem Nursery is located in Jumeirah. Together the institutions enroll almost 1500 students aged 3 to 18.:EducationalInstitution}}

{{ Martin Marty McKinnon (born 5 July 1975 in Adelaide) is a former Australian rules footballer who played with Adelaide Geelong and the Brisbane Lions in the Australian Football League (AFL).McKinnon was recruited by Adelaide in the 1992 AFL Draft with their first ever national draft pick. He was the youngest player on Adelaide's list at the time and played for Central District in the SANFL when not appearing with Adelaide.:Athlete}}

{{ The Wedell-Williams XP-34 was a fighter aircraft design submitted to the United States Army Air Corps (USAAC) before World War II by Marguerite Clark Williams widow of millionaire Harry P. Williams former owner and co-founder of the Wedell-Williams Air Service Corporation.:MeanOfTransportation}}

{{"{}":'''.format(example['text'])
        
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
            fc_vals.append(output[2][0][0].squeeze())
            
        # else:
        #     print(f"Text: {example['text']}, True label: {true_label_text}, Predicted label: {predicted_label}")
        
        label_tok = tokenizer.encode(true_label_text)[1]
        
        # Update progress bar description with current accuracy
        progress_bar.set_description(f"Accuracy: {round(correct/i,3)*100}%")
        confidence += output[1][0][0][label_tok].item()
        
        
        

    return round(correct/i,4), round(confidence/i,4), fc_vals

import torch
from torch import nn
from typing import Tuple

def manual_generate_llma_batch_insert(
    model: nn.Module,
    input_ids: torch.LongTensor,          # (B, Lₘᵢₙ ≤ … ≤ Lₘₐₓ)
    attention_mask: torch.LongTensor,     # (B, Lₘₐₓ)
    orig_lens: torch.LongTensor,          # (B,)
    max_new_tokens: int = 5,
) -> Tuple[torch.LongTensor, torch.FloatTensor, torch.FloatTensor]:
    """
    Same signature as before, but the final tensor keeps the shape
    (B, max_orig_len + max_new_tokens) and new tokens are *inserted*
    at orig_len + step rather than tacked to the end.
    """

    device        = input_ids.device
    B             = input_ids.size(0)
    pad_token_id  = getattr(model.config, "pad_token_id", 0)

    max_orig_len  = input_ids.size(1)                      # already padded to this
    full_len      = max_orig_len + max_new_tokens          # final width

    # 1) Pre‑allocate padded canvases
    generated = torch.full((B, full_len),
                           pad_token_id,
                           dtype=input_ids.dtype,
                           device=device)
    generated[:, :max_orig_len] = input_ids

    full_mask = torch.zeros_like(generated, dtype=attention_mask.dtype)
    full_mask[:, :max_orig_len] = attention_mask

    # Vector pointing at the *next* free slot for each row
    insert_pos = orig_lens.clone()                         # (B,)

    # -------------------------------------------------- #
    # Step‑0: run once on the original context
    # -------------------------------------------------- #
    with torch.no_grad(), nethook.TraceDict(model, ['model.mask_layer']) as ret:
        logits0 = model(
            input_ids      = input_ids,
            attention_mask = attention_mask
        ).logits                                            # (B, Lₘₐₓ, vocab)

        next_logits  = logits0[torch.arange(B), orig_lens-1, :]   # (B, vocab)
        step0_conf   = torch.softmax(next_logits, dim=-1).cpu()   # (B, vocab)
        next_tokens  = torch.argmax(next_logits, dim=-1)          # (B,)

        # FC activations at the last real token
        per_layer = [
            layer_out[torch.arange(B, device=device), orig_lens-1, :].cpu()
            for layer_out in (ret[k].output for k in ret)
        ]
        step0_fc_vals = torch.stack(per_layer, dim=1)       # (B, n_layers, hidden)

    # Insert step‑0 tokens
    generated[torch.arange(B), insert_pos]  = next_tokens
    full_mask[torch.arange(B), insert_pos]  = 1
    insert_pos += 1                                         # advance pointers

    # --------------------------------------------- #
    # Steps 1 .. max_new_tokens‑1
    # --------------------------------------------- #
    for _ in range(max_new_tokens - 1):
        # Slice only up to the furthest used column for a tiny speed win
        active_len = insert_pos.max().item()                # int
        with torch.no_grad():
            logits = model(
                input_ids      = generated[:, :active_len],
                attention_mask = full_mask[:, :active_len]
            ).logits                                         # (B, active_len, vocab)
            last_pos    = insert_pos - 1
            next_logits = logits[
            torch.arange(B, device=device),last_pos,:]
            next_tokens = torch.argmax(next_logits, dim=-1) # (B,)

        # Insert at current pointers
        generated[torch.arange(B), insert_pos] = next_tokens
        full_mask[torch.arange(B), insert_pos] = 1
        insert_pos += 1

    return generated, step0_conf, step0_fc_vals








# --------------------------------------------------------------------------- #
#                   Evaluation loop (classification)                          #
# --------------------------------------------------------------------------- #
def evaluate_llma_classification_batch(model, eval_dataset, tokenizer,
                                       batch_size: int = 8,
                                       device: str = "cuda",
                                       max_new_tokens: int = 5):
    label_mapping = {
        0:'Company', 1:'EducationalInstitution', 2:'Artist', 3:'Athlete',
        4:'OfficeHolder', 5:'MeanOfTransportation', 6:'Building',
        7:'NaturalPlace', 8:'Village', 9:'Animal', 10:'Plant',
        11:'Album', 12:'Film', 13:'WrittenWork'
    }

    model.to(device).eval()
    total, correct, total_conf = 0, 0, 0.0
    fc_vals_correct = []

    loop = tqdm(range(0, len(eval_dataset), batch_size), desc="Batches")

    for start in loop:
        batch = eval_dataset[start : start + batch_size]
        texts  = batch["text"]
        labels = batch["label"]

        # ------------- build prompts -----------------------------------------
        prompts, gold_labels = [], []
        for text, lab_i in zip(texts, labels):
            

            gold = label_mapping[int(lab_i)]            # map int -> string
            gold_labels.append(gold)
            prompts.append(
                '''Choose from one of these categories: Company, EducationalInstitution, Artist, Athlete, OfficeHolder, MeanOfTransportation, Building, NaturalPlace, Village, Animal, Plant, Album, Film, WrittenWork. Be careful distinguishing between similar categories.

{{ Abbott of Farnham E D Abbott Limited was a British coachbuilding business based in Farnham Surrey trading under that name from 1929. A major part of their output was under sub-contract to motor vehicle manufacturers. Their business closed in 1972.:Company}}

{{ Dubai Gem Private School (DGPS) is a British school located in the Oud Metha area of Dubai United Arab Emirates. Dubai Gem Nursery is located in Jumeirah. Together the institutions enroll almost 1500 students aged 3 to 18.:EducationalInstitution}}

{{ Martin Marty McKinnon (born 5 July 1975 in Adelaide) is a former Australian rules footballer who played with Adelaide Geelong and the Brisbane Lions in the Australian Football League (AFL).McKinnon was recruited by Adelaide in the 1992 AFL Draft with their first ever national draft pick. He was the youngest player on Adelaide's list at the time and played for Central District in the SANFL when not appearing with Adelaide.:Athlete}}

{{ The Wedell-Williams XP-34 was a fighter aircraft design submitted to the United States Army Air Corps (USAAC) before World War II by Marguerite Clark Williams widow of millionaire Harry P. Williams former owner and co-founder of the Wedell-Williams Air Service Corporation.:MeanOfTransportation}}

{{"{}":'''.format(text)
            )

        enc = tokenizer(prompts, padding=True, return_tensors='pt').to(device)
        orig_lens = enc['attention_mask'].sum(dim=1)          # (B,)

        # ------------- generation + FC capture -------------------------------
        (gen, step0_conf, step0_fc) = manual_generate_llma_batch_insert(
            model,
            enc['input_ids'], enc['attention_mask'],
            orig_lens,
            max_new_tokens=max_new_tokens
        )

        # ------------- post‑process ------------------------------------------
        for i, gold in enumerate(gold_labels):
            gen_text = tokenizer.decode(gen[i, orig_lens[i]:], skip_special_tokens=True)  # strip context
            
            pred = gen_text.split('}')[0].strip('"').strip()
            # print("--------",pred)
            total += 1
            if pred == gold:
                correct += 1
                fc_vals_correct.append(step0_fc[i].squeeze())            # (n_layers, hid)
            else:
                print("Original: ", gold)
                print("Predicted: ", pred)
                print("Gen_text", gen_text)

            # confidence for gold label token in first step
            gold_tok = tokenizer.encode(gold, add_special_tokens=False)[0]
            total_conf += step0_conf[i, gold_tok].item()

        loop.set_description(f"Accuracy: {100*correct/total:.1f}%")

    accuracy   = round(correct / total, 4)
    mean_conf  = round(total_conf / total, 4)
    fc_vals    = torch.stack(fc_vals_correct) if fc_vals_correct else torch.tensor([])

    return accuracy, mean_conf, fc_vals


















from rouge_score import rouge_scorer
def evaluate_llma_summarization(model, eval_dataset, tokenizer, max_samples=100):
    # Take a sample of the dataset
    if len(eval_dataset) > max_samples:
        eval_dataset = eval_dataset.select(range(max_samples))
        
    progress_bar = tqdm(range(len(eval_dataset)), desc="Processing examples")
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    fc_vals = []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    model.to('cuda')
    model.eval()
    
    for i in progress_bar:
        example = eval_dataset[i]  # Get the example by index
        prompt = """
Example:
Article: Facebook page supporting tradition gains one million 'likes' in a day. 'Don't let the Netherlands' most beautiful tradition disappear,' it says. UN has condemned the tradition claiming it reflects racial prejudice.
Summary: Facebook page supporting controversial Dutch tradition gains massive support despite UN criticism.

Now generate a short summary for the following:
Article:
{text}
Summary:
""".format(text=example['article'][:1000])  # Use article field from CNN/Daily Mail
        
        input_ids = tokenizer([prompt, prompt], return_tensors='pt')
        
        # Generate output
        output = manual_generate_llma(
            model,
            input_ids['input_ids'].to('cuda'),
            input_ids['attention_mask'].to('cuda'),
            input_ids['input_ids'].shape[1] + 150
        )
        
        # Get predicted summary
        predicted_summary = tokenizer.decode(
            output[0][0][input_ids['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        # Calculate ROUGE scores
        scores = scorer.score(example['highlights'], predicted_summary)
        
        for metric in rouge_scores:
            rouge_scores[metric].append(scores[metric].fmeasure)
        
        fc_vals.append(output[2][0][0].squeeze())
        
        # Update progress bar with current ROUGE-1 score
        avg_rouge1 = np.mean(rouge_scores['rouge1'])
        progress_bar.set_description(f"ROUGE-1: {avg_rouge1:.4f}")
        
    # Calculate average scores
    avg_scores = {metric: np.mean(values) for metric, values in rouge_scores.items()}
    
    return avg_scores, fc_vals




def evaluate_llma_nli(model, eval_dataset, tokenizer, max_samples=100):
    if len(eval_dataset) > max_samples:
        eval_dataset = eval_dataset.select(range(max_samples))

    progress_bar      = tqdm(range(len(eval_dataset)), desc="Processing examples")
    gold_labels       = []
    pred_labels       = []
    fc_vals_correct   = []          # vectors at the class‑token position

    model.to("cuda").eval()
    id2lbl = {0: "entailment", 1: "neutral", 2: "contradiction"}

    for idx in progress_bar:
        ex = eval_dataset[idx]
        prompt = FEW_SHOT.format(premise=ex["premise"].strip(),
                                 hypothesis=ex["hypothesis"].strip())

        inputs      = tokenizer([prompt, prompt], return_tensors="pt")
        prompt_len  = inputs["input_ids"].shape[1]

        out = manual_generate_llma(
                  model,
                  inputs["input_ids"].to("cuda"),
                  inputs["attention_mask"].to("cuda"),
                  max_new_tokens=5
              )

        gen_ids       = out[0][0][prompt_len:]              # generated ids only
        first_tok_id  = gen_ids[0].item()                   # id of class token
        gen_text      = tokenizer.decode(first_tok_id).lower().strip()

        # map to label
        if gen_text.startswith("entail"):
            pred = "entailment"
        elif gen_text.startswith("contrad"):
            pred = "contradiction"
        else:
            pred = "neutral"

        gold = id2lbl.get(ex["label"], ex["label"])
        pred_labels.append(pred)
        gold_labels.append(gold)

        # -------- take fc at the position of that first generated token -------
        class_vec = out[2][0][prompt_len]   # [batch=0, seq‑pos=prompt_len, hidden]
        if pred == gold:
            fc_vals_correct.append(class_vec.squeeze())

        progress_bar.set_description(
            f"ACC: {accuracy_score(gold_labels, pred_labels):.4f}"
        )

    # aggregate metrics
    acc  = accuracy_score(gold_labels, pred_labels)
    prec, rec, f1, _ = precision_recall_fscore_support(
                           gold_labels, pred_labels, average="macro", zero_division=0
                       )

    metrics = {"accuracy": acc,
               "macro_precision": prec,
               "macro_recall": rec,
               "macro_f1": f1}

    return metrics, fc_vals_correct




from collections import Counter
import re

def evaluate_llma_squad(model, eval_dataset, tokenizer, max_samples=50):
    """
    Evaluate the model on the SQuAD question answering task.
    
    Args:
        model: The model to evaluate
        eval_dataset: The SQuAD dataset to evaluate on
        tokenizer: The tokenizer to use
        max_samples: Maximum number of samples to evaluate
    
    Returns:
        scores: Dictionary containing EM (Exact Match) and F1 scores
    """
    # Take a sample of the dataset
    if len(eval_dataset) > max_samples:
        eval_dataset = eval_dataset.select(range(max_samples))
    
    progress_bar = tqdm(range(len(eval_dataset)), desc="Evaluating QA")
    exact_matches = []
    f1_scores = []
    
    # Get device
    device = next(model.parameters()).device
    
    model.eval()
    
    for i in progress_bar:
        try:
            # Get the example by index
            example = eval_dataset[i]
            
            # Format prompt for question answering
            prompt = f"""Context: {example['context']}

Question: {example['question']}

Answer:"""
            
            # Tokenize input
            inputs = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate answer
            with torch.no_grad():
                output = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=50,
                    num_beams=1,
                    early_stopping=True
                )
            
            # Decode generated answer
            predicted_answer = tokenizer.decode(
                output[0, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            # Get ground truth answer
            ground_truth = example['answers']['text'][0]  # Using first answer as reference
            
            # Calculate metrics
            exact_match = compute_exact_match(predicted_answer, ground_truth)
            f1 = compute_f1_score(predicted_answer, ground_truth)
            
            exact_matches.append(exact_match)
            f1_scores.append(f1)
            
            # Update progress bar
            avg_em = np.mean(exact_matches)
            avg_f1 = np.mean(f1_scores)
            progress_bar.set_description(f"EM: {avg_em:.4f}, F1: {avg_f1:.4f}")
        
        except Exception as e:
            print(f"Error processing example {i}: {e}")
            continue
    
    # Calculate final scores
    scores = {
        "exact_match": np.mean(exact_matches) if exact_matches else 0.0,
        "f1": np.mean(f1_scores) if f1_scores else 0.0
    }
    
    return scores

def compute_exact_match(prediction, ground_truth):
    """
    Calculate exact match score (1 if prediction matches ground truth exactly, 0 otherwise)
    """
    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    return float(prediction == ground_truth)

def compute_f1_score(prediction, ground_truth):
    """
    Calculate word-level F1 score between prediction and ground truth
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    
    # Edge cases
    if len(ground_truth_tokens) == 0 or len(prediction_tokens) == 0:
        return int(ground_truth_tokens == prediction_tokens)
    
    # Count common tokens
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_common = sum(common.values())
    
    # Edge case - both empty
    if num_common == 0:
        return 0
    
    precision = num_common / len(prediction_tokens)
    recall = num_common / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1

def normalize_answer(text):
    """
    Normalize answer text for comparison (lowercase, remove articles, punctuation, etc.)
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove articles and punctuation
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text






def evaluate_llma_language_modeling(model, eval_dataset, tokenizer, max_samples=50):
    """
    Evaluate the perplexity of the model on a language modeling task.
    
    Args:
        model: The model to evaluate
        eval_dataset: The dataset to evaluate on
        tokenizer: The tokenizer to use
        max_samples: Maximum number of samples to evaluate
    
    Returns:
        perplexity: The average perplexity across samples
    """
    # Take a sample of the dataset
    if len(eval_dataset) > max_samples:
        eval_dataset = eval_dataset.select(range(max_samples))
    
    progress_bar = tqdm(range(len(eval_dataset)), desc="Evaluating LM")
    total_loss = 0
    total_tokens = 0
    
    # Get device
    device = next(model.parameters()).device
    
    model.eval()
    
    for i in progress_bar:
        # Get the example by index
        example = eval_dataset[i]
        
        # Tokenize input
        inputs = tokenizer(example['text'], return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
        total_loss += loss.item() * inputs["input_ids"].size(1)
        total_tokens += inputs["input_ids"].size(1)
        
        # Update progress bar with current perplexity
        current_perplexity = torch.exp(torch.tensor(total_loss / total_tokens)).item()
        progress_bar.set_description(f"Perplexity: {current_perplexity:.4f}")
    
    # Calculate perplexity
    perplexity = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    
    return {"perplexity": perplexity}

def reset_llma(model):
    model.model.mask_layer.lower_bound = torch.tensor(float('inf')).to(device)
    model.model.mask_layer.upper_bound = torch.tensor(float('-inf')).to(device)
    return model


def mask_range_llma(model, mask, fc_vals, tao):
    fc_vals = np.array(fc_vals)
    mean = torch.tensor(np.mean(fc_vals, axis=0))
    std = torch.tensor(np.std(fc_vals, axis=0))
    mask = mask.to(torch.bool)
    
    lower_bound = torch.full_like(mean, torch.inf)
    lower_bound[~mask] = mean[~mask] - tao*std[~mask]
    upper_bound = torch.full_like(mean, -torch.inf)
    upper_bound[~mask] = mean[~mask] + tao*std[~mask]
    
    model.model.mask_layer.lower_bound = lower_bound.to(device)
    model.model.mask_layer.upper_bound = upper_bound.to(device)
    # model.model.mask_layer.replacement_values = mean.to(device)
    
    return model