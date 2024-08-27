import os
from tqdm import tqdm_notebook as tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
from utilities import train, eval, pad, compute_masks, mask
from POS_dataset import PosDataset

import nltk
tagged_sents = nltk.corpus.treebank.tagged_sents()

tags = list(set(word_pos[1] for sent in tagged_sents for word_pos in sent))

",".join(tags)

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
        return enc, logits, y, y_hat
    
    
model = Net(vocab_size=len(tag2idx))
model.to(device)

train_dataset = PosDataset(train_data, tokenizer, tag2idx)
eval_dataset = PosDataset(test_data, tokenizer, tag2idx)

train_iter = data.DataLoader(dataset=train_dataset,
                             batch_size=8,
                             shuffle=True,
                             num_workers=1,
                             collate_fn=pad)
test_iter = data.DataLoader(dataset=eval_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=1,
                             collate_fn=pad)
activation_iter = data.DataLoader(dataset=train_dataset+eval_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=1,
                             collate_fn=pad)

optimizer = optim.Adam(model.parameters(), lr = 0.0001)

criterion = nn.CrossEntropyLoss(ignore_index=0)


train(model, train_iter, optimizer, criterion)
enc_dict = eval(model, activation_iter, idx2tag, tag2idx)

mask_max, mask_std = compute_masks(enc_dict[2],0.5)

# model = mask(model,mask_max)

enc_dict = eval(model, activation_iter, idx2tag, tag2idx)

# # size of encodings_by_tag
# i=0
# for k, v in enc_dict.items():
#     print(i,k ,idx2tag[k], len(v))
#     i+=1
    
# print(idx2tag)