import torch, gc
gc.collect()
torch.cuda.empty_cache()

import pandas as pd
import os
import string
import numpy as np
import glob

from torch.utils.data.dataset import Dataset

captions_file = 'D:/data/vqa/coco/simple_vqa/caption_sample.txt'
df = pd.read_csv(captions_file)

# print(df["captions"])

class TextGeneration(Dataset):
    def clear_text(self, txt):
        txt = "".join(v for v in txt if v not in string.punctuation).lower()
        return txt

        ##데이터 불러와 BOW 생성

    def __init__(self):

        all_cation = [h for h in df["captions"] if h != "Unknown"]

        self.corpus = [self.clear_text(x) for x in all_cation]
        self.BOW = {}

        for line in self.corpus:
            for word in line.split():
                if word not in self.BOW.keys():
                    self.BOW[word] = len(self.BOW.keys())

        self.data = self.generate_sequence(self.corpus)\

        ##BOW이용한 시계열 구성

    def generate_sequence(self, txt):
        seq = []

        for line in txt:
            line = line.split()
            line_bow = [self.BOW[word] for word in line]

            data = [([line_bow[i], line_bow[i + 1]], line_bow[i + 2])
                    for i in range(len(line_bow) - 2)]

            seq.extend(data)

        return seq

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data = np.array(self.data[i][0])
        label = np.array(self.data[i][1]).astype(np.float32)

        return data, label

import torch.nn as nn
import torch

class LSTM(nn.Module):
    def __init__(self,num_embeddings):
        super(LSTM, self).__init__()

        self.embed = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=16)

        self.lstm = nn.LSTM(input_size=16,hidden_size=64,num_layers=5)

        self.fc1 = nn.Linear(128,num_embeddings)
        self.fc2 = nn.Linear(num_embeddings,num_embeddings)
        self.relu = nn.ReLU()


    def forward(self,x):
        x = self.embed(x)

        x,_ = self.lstm(x)
        x = torch.reshape(x,(x.shape[0],-1))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


##모델 학습

import tqdm

from torch.utils.data.dataloader import DataLoader
from torch.optim.adam import Adam

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = TextGeneration()

model = LSTM(num_embeddings=len(dataset.BOW)).to(device)
loader = DataLoader(dataset,batch_size=64)
optim = Adam(model.parameters(),lr = 0.001)

for epoch in range(200):
    iterator = tqdm.tqdm(loader)
    for data,label in iterator:

        optim.zero_grad()

        pred = model(torch.tensor(data,dtype=torch.long).to(device))

        loss = nn.CrossEntropyLoss()(pred,torch.tensor(label,dtype=torch.long).to(device))

        optim.zero_grad()
        loss.backward()
        optim.step()

        iterator.set_description(f"epoch{epoch} loss:{loss.item()}")

torch.save(model.state_dict(),"lstm.pth")

def generate(model,BOW,string="the skateboarder ",strlen=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"input word:{string}")

    with torch.no_grad():
        for p in range(strlen):
            words = torch.tensor([BOW[w] for w in string.split()],dtype=torch.long).to(device)

            input_tensor = torch.unsqueeze(words[-2:],dim = 0)
            output = model(input_tensor)
            out_word = torch.argmax(output).cpu().numpy()
            string+= list(BOW.keys())[out_word]
            string+=" "

        print(f"predicted sentence:{string}")


model.load_state_dict(torch.load("lstm.pth",map_location=device))
pred = generate(model,dataset.BOW)