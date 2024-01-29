import os
import random
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

import evaluate
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer

from sklearn.model_selection import train_test_split


class BERTDataset(Dataset):
    def __init__(self, data, tokenizer):
        input_texts = data['text']
        targets = data['target']
        self.inputs = []
        self.labels = []
        
        for text, label in zip(input_texts, targets):
            tokenized_input = tokenizer(text, padding='max_length', truncation=True, return_tensors='pt')
            self.inputs.append(tokenized_input)
            self.labels.append(torch.tensor(label))
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx]['input_ids'].squeeze(0),  
            'attention_mask': self.inputs[idx]['attention_mask'].squeeze(0),
            'labels': self.labels[idx].squeeze(0)
        }
    
    def __len__(self):
        return len(self.labels)


SEED = 456
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, '/data')
OUTPUT_DIR = os.path.join(BASE_DIR, '/output')

data = pd.read_csv('./data/train.csv')
dataset_train, dataset_valid = train_test_split(data, test_size=0.3, stratify=data['target'],random_state=SEED)

model_name = 'klue/bert-base'
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=7).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(model_name)


data_train = BERTDataset(dataset_train, tokenizer)
data_valid = BERTDataset(dataset_valid, tokenizer)

n,m=len(data_train),len(data_train[0]['input_ids'])
torch_embed,torch_label=torch.zeros(n, m),torch.zeros(n, 1)
for i in range(len(data_train)):
    torch_embed[i],torch_label[i]=data_train[i]['input_ids'],data_train[i]['labels']


n_components=3

from sklearn.manifold import TSNE

model = TSNE(n_components=n_components)
TSNEembedded = model.fit_transform(torch_embed)
df = pd.DataFrame(data=TSNEembedded, columns = ['pc1', 'pc2', 'pc3'])
df['target']=torch_label

import seaborn as seaborn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

seaborn.set_style("darkgrid")

fig = plt.figure(figsize=(6,6))

axes = Axes3D(fig)
x,y,z=df['pc1'].tolist(),df['pc2'].tolist(),df['pc3'].tolist()
co=df['target'].tolist()
axes.scatter(x, y, z, c=co, marker='o')

axes.set_xlabel("x")
axes.set_ylabel("y")
axes.set_zlabel("z")
plt.show()

