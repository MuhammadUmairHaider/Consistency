__author__ = "kyubyong"
__address__ = "https://github.com/kyubyong/nlp_made_easy"
__email__ = "kbpark.linguist@gmail.com"

import os
from tqdm import tqdm_notebook as tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim

import nltk
tagged_sents = nltk.corpus.treebank.tagged_sents()
len(tagged_sents)

tags = list(set(word_pos[1] for sent in tagged_sents for word_pos in sent))

",".join(tags)

# By convention, the 0'th slot is reserved for padding.
tags = ["<pad>"] + tags

tag2idx = {tag:idx for idx, tag in enumerate(tags)}
idx2tag = {idx:tag for idx, tag in enumerate(tags)}

# Let's split the data into train and test (or eval)
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(tagged_sents, test_size=.1)
len(train_data), len(test_data)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline
import torch

model_name = "QCRI/bert-base-multilingual-cased-pos-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)

class PosDataset(data.Dataset):
    def __init__(self, tagged_sents):
        sents, tags_li = [], [] # list of lists
        for sent in tagged_sents:
            words = [word_pos[0] for word_pos in sent]
            tags = [word_pos[1] for word_pos in sent]
            sents.append(["[CLS]"] + words + ["[SEP]"])
            tags_li.append(["<pad>"] + tags + ["<pad>"])
        self.sents, self.tags_li = sents, tags_li

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        words, tags = self.sents[idx], self.tags_li[idx] # words, tags: string list

        # We give credits only to the first piece.
        x, y = [], [] # list of ids
        is_heads = [] # list. 1: the token is the first piece of a word
        for w, t in zip(words, tags):
            tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
            xx = tokenizer.convert_tokens_to_ids(tokens)

            is_head = [1] + [0]*(len(tokens) - 1)

            t = [t] + ["<pad>"] * (len(tokens) - 1)  # <PAD>: no decision
            yy = [tag2idx[each] for each in t]  # (T,)

            x.extend(xx)
            is_heads.extend(is_head)
            y.extend(yy)

        assert len(x)==len(y)==len(is_heads), "len(x)={}, len(y)={}, len(is_heads)={}".format(len(x), len(y), len(is_heads))

        # seqlen
        seqlen = len(y)

        # to string
        words = " ".join(words)
        tags = " ".join(tags)
        return words, x, is_heads, tags, y, seqlen


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

class Net(nn.Module):
    def __init__(self, vocab_size=None):
        super().__init__()
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.bert = self.model.bert
        self.masking_layer = torch.ones(768).to("cuda")

        self.fc = nn.Linear(768, vocab_size)
        self.device = device

    def forward(self, x, y):
        '''
        x: (N, T). int64
        y: (N, T). int64
        '''
        x = x.to(device)
        y = y.to(device)
        
        if self.training:
            self.bert.train()
            encoded_layers = self.bert(x)
            enc = encoded_layers[-1]
        else:
            self.bert.eval()
            with torch.no_grad():
                encoded_layers = self.bert(x)
                enc = encoded_layers[-1]
        # enc = nn.ReLU(enc)
        enc = enc * self.masking_layer
        logits = self.fc(enc)
        y_hat = logits.argmax(-1)
        confidence = logits.softmax(-1).max(-1).values
        return enc, logits, y, y_hat, confidence
    
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
            
model = Net(vocab_size=len(tag2idx))
model.to(device)
# model = nn.DataParallel(model)


train_dataset = PosDataset(train_data)
eval_dataset = PosDataset(test_data)

train_iter = data.DataLoader(dataset=train_dataset,
                             batch_size=8,
                             shuffle=True,
                             num_workers=8,
                             collate_fn=pad)
test_iter = data.DataLoader(dataset=eval_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=8,
                             collate_fn=pad)

optimizer = optim.Adam(model.parameters(), lr = 0.0001)

criterion = nn.CrossEntropyLoss(ignore_index=0)


def eval(model, iterator):
    model.eval()

    Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []
    encodings_by_tag = {}  # Dictionary to store encodings by tag
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_heads, tags, y, seqlens = batch

            enc, _, _, y_hat, conf = model(x, y)  # y_hat: (N, T)

            Words.extend(words)
            Is_heads.extend(is_heads)
            Tags.extend(tags)
            Y.extend(y.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())
            for t, encoding, h in zip(y[0].numpy(), enc[0], is_heads[0]):
                if(h):
                    if t not in encodings_by_tag:
                        encodings_by_tag[t] = []
                
                    encodings_by_tag[t].append(encoding.cpu().numpy())

    ## gets results and save
    with open("result", 'w') as fout:
        for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
            y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
            preds = [idx2tag[hat] for hat in y_hat]
            assert len(preds)==len(words.split())==len(tags.split())
            for w, t, p in zip(words.split()[1:-1], tags.split()[1:-1], preds[1:-1]):
                fout.write("{} {} {}\n".format(w, t, p))
            fout.write("\n")
            
    ## calc metric
    y_true =  np.array([tag2idx[line.split()[1]] for line in open('result', 'r').read().splitlines() if len(line) > 0])
    y_pred =  np.array([tag2idx[line.split()[2]] for line in open('result', 'r').read().splitlines() if len(line) > 0])

    acc = (y_true==y_pred).astype(np.int32).sum() / len(y_true)

    print("acc=%.3f"%acc)
    return encodings_by_tag


for i in range(20):
    train(model, train_iter, optimizer, criterion)
# enc_dict = eval(model, test_iter)


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
    mask_intersection = torch.logical_or(mask_std, mask_max).float()
    
    return mask_max, mask_std, mask_intersection, mask_max_low_std

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
    bottom_50_percent_std_count = int(0.7 * len(std_vals))
    bottom_50_percent_std_indices = torch.argsort(std_vals)[:bottom_50_percent_std_count]
    
    # Create a mask for bottom 50% std values
    bottom_50_percent_std_mask = torch.zeros_like(std_vals, dtype=torch.bool)
    bottom_50_percent_std_mask[bottom_50_percent_std_indices] = True
    
    # Filter mean values
    mean_vals_filtered = mean_vals.clone()
    mean_vals_filtered[~bottom_50_percent_std_mask] = float('-inf')
    
    # Compute mask
    return compute_max_mask(mean_vals_filtered, percent)


from utilities import mask, eval
for tok in range(45):
    print(idx2tag[tok],"----------------")
    model.masking_layer = torch.ones(768).to("cuda")
    activation_iter = data.DataLoader(dataset=train_dataset+eval_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=1,
                                collate_fn=pad)
    print("Original:")
    enc_dict = eval(model, activation_iter, idx2tag, tag2idx, tok)

    mask_max, mask_std, mask_intersection,mask_max_low_std = compute_masks(enc_dict[tok],0.3)
    print("STD:")
    model = mask(model,mask_max_low_std)

    enc_dict = eval(model, activation_iter, idx2tag, tag2idx, tok)
    print("Max:")
    model = mask(model,mask_max)

    enc_dict = eval(model, activation_iter, idx2tag, tag2idx, tok)
    
    print("-----------------------------")
    

# # size of encodings_by_tag
# i=0
# for k, v in enc_dict.items():
#     print(i,k ,idx2tag[k], len(v))
#     i+=1
    
# print(idx2tag)