import torch
import torch.nn as nn
import torch.optim as optim

import math
import numpy as np
import pandas as pd
import random
import re
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

class Trasformer(nn.Module):
    def __init__(self,num_tokens,dim_model,num_heads,num_encoder_layers,num_decoder_layers,dropout_p):
        super().__init__()


        self.model_type = "Transformer"
        self.dim_model = dim_model
        self.positional_encoder = PositionalEncoding(dim_model = dim_model, dropput_p=dropout_p,max_len=5000)
        self.embedding = nn.Embedding(num_tokens,dim_model)
        self.transformer = nn.Transformer(
            d_model=dim_model, #트랜스포머의 인코더와 디코더에서의 정해진 입력과 출력의 크기(default = 512)
            nhead=num_heads, # 멀티헤드어텐션 모델의 헤드 수 , 어텐션을 사용할때 여러개로 분할해서
                             #병렬로 어텐션을 수행하고 결과값을 다시 하나로 합치는 방식에서 병렬의 수 default = 8
            num_encoder_layers=num_encoder_layers, #인코더 층
            num_decoder_layers=num_decoder_layers, #디코더 층
            dropout=dropout_p,
        )
        #멀티헤드 어텐션과 피드 포워드 등을 처리
        #Linear, Postional Encoding, Embedding 은 만들어줘야함
        self.out = nn.Linear(dim_model,num_tokens)

    def forward(self,src,tgt,tgt_mask=None,src_pad_mask=None,tgt_pad_mask=None):
        src = self.embedding(src)*math.sqrt(self.dim_model)
        tgt = self.embedding(tgt) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        src = src.permute(1,0,2)
        tgt = tgt.permute(1,0,2)

        transformer_out = self.transformer(src,tgt,tgt_mask=tgt_mask,src_key_padding_mask=src_pad_mask,tgt_key_padding_mask=tgt_pad_mask)
        out = self.out(transformer_out)

        return out



    def get_tgt_mask(self,size)->torch.tensor:
        mask = torch.tril(torch.ones(size,size)==1)
        mask = mask.float()
        mask = mask.masked_fill(mask ==0, float('-inf'))
        mask = mask.masked_fill(mask == 0, float(0.0))

        return mask

    def create_pad_mask(self,matrix:torch.tensor,pad_token:int)->torch.tensor:
        return (matrix == pad_token)

class PositionalEncoding(nn.Module):
    def __init__(self,dim_model,dropput_p,max_len):
        super().__init__()
        self.dropout = nn.Dropout(dropput_p)
        pos_encoding = torch.zeros(max_len,dim_model)
        position_list = torch.arange(0,max_len,dtype=torch.float).view(-1,1)
        division_term = torch.exp(torch.arange(0,dim_model,2).float()*(-math.log(10000.0)/dim_model))
        pos_encoding[:,0::2] = torch.sin(position_list * division_term)
        pos_encoding[:,1::2] = torch.cos(position_list * division_term)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0,1)
        self.register_buffer("pos_encoding",pos_encoding)

    def forward(self,token_embedding:torch.tensor)->torch.tensor:
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0),:])

def generate_random_data(n):
    SOS_token = np.array([2])
    EOS_token = np.array([3])
    length = 8

    data = []

    for i in range(n//3):
        x = np.concatenate((SOS_token,np.ones(length),EOS_token))
        y = np.concatenate((SOS_token,np.ones(length),EOS_token))
        data.append([x,y])

    for i in range(n//3):
        x = np.concatenate((SOS_token,np.zeros(length),EOS_token))
        y = np.concatenate((SOS_token,np.zeros(length),EOS_token))
        data.append([x,y])


    # 1,0,1,0 -> 1,0,1,0,1
    for i in range(n // 3):
        x = np.zeros(length)
        start = random.randint(0, 1)

        x[start::2] = 1

        y = np.zeros(length)
        if x[-1] == 0:
            y[::2] = 1
        else:
            y[1::2] = 1

        x = np.concatenate((SOS_token, x, EOS_token))
        y = np.concatenate((SOS_token, y, EOS_token))
        data.append([x, y])

    np.random.shuffle(data)

    return data


#크기가 16인 배치 형태로 만들어 줍니다.
def batchify_data(data, batch_size=16, padding=False, padding_token=-1):
    batches = []
    for idx in range(0, len(data), batch_size):
        # batch_size 크기가 아닌 경우 마지막 비트를 얻지 않도록 합니다.
        if idx + batch_size < len(data):
            # 여기서 배치의 최대 길이를 가져와 PAD 토큰으로 길이를 정규화해야 합니다.
            if padding:
                max_batch_length = 0
                # batch에서 가장 긴 문장 가져오기
                for seq in data[idx : idx + batch_size]:
                    if len(seq) > max_batch_length:
                        max_batch_length = len(seq)

                # 최대 길이에 도달할 때까지 X 패딩 토큰을 추가합니다.
                for seq_idx in range(batch_size):
                    remaining_length = max_batch_length - len(data[idx + seq_idx])
                    data[idx + seq_idx] += [padding_token] * remaining_length

            batches.append(np.array(data[idx : idx + batch_size]).astype(np.int64))

    print(f"{len(batches)} batches of size {batch_size}")

    return batches


train_data = generate_random_data(9000)
val_data = generate_random_data(3000)

train_dataloader = batchify_data(train_data)
val_dataloader = batchify_data(val_data)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Trasformer(num_tokens=4, dim_model=8, num_heads=2, num_encoder_layers=3, num_decoder_layers=3, dropout_p=0.1).to(device)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

def train_loop(model, opt, loss_fn, dataloader):
    model.train()
    total_loss = 0

    for batch in dataloader:
        X, y = batch[:, 0], batch[:, 1]
        X, y = torch.tensor(X).to(device), torch.tensor(y).to(device)

        # 이제 tgt를 1만큼 이동하여 <SOS>를 사용하여 pos 1에서 토큰을 예측합니다.
        y_input = y[:,:-1]
        y_expected = y[:,1:]

        # 다음 단어를 마스킹하려면 마스크 가져오기
        sequence_length = y_input.size(1)
        tgt_mask = model.get_tgt_mask(sequence_length).to(device)

        # X, y_input 및 tgt_mask를 전달하여 표준 training
        pred = model(X, y_input, tgt_mask)

        # Permute 를 수행하여 batch size 가 처음이 되도록
        pred = pred.permute(1, 2, 0)
        loss = loss_fn(pred, y_expected)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.detach().item()

    return total_loss / len(dataloader)


def validation_loop(model, loss_fn, dataloader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            X, y = batch[:, 0], batch[:, 1]
            X, y = torch.tensor(X, dtype=torch.long, device=device), torch.tensor(y, dtype=torch.long, device=device)

            y_input = y[:,:-1]
            y_expected = y[:,1:]

            sequence_length = y_input.size(1)
            tgt_mask = model.get_tgt_mask(sequence_length).to(device)

            pred = model(X, y_input, tgt_mask)

            pred = pred.permute(1, 2, 0)
            loss = loss_fn(pred, y_expected)
            total_loss += loss.detach().item()

    return total_loss / len(dataloader)


def fit(model, opt, loss_fn, train_dataloader, val_dataloader, epochs):
    # 나중에 plotting 하기위해
    train_loss_list, validation_loss_list = [], []

    print("Training and validating model")
    for epoch in range(epochs):
        print("-"*25, f"Epoch {epoch + 1}","-"*25)

        train_loss = train_loop(model, opt, loss_fn, train_dataloader)
        train_loss_list += [train_loss]

        validation_loss = validation_loop(model, loss_fn, val_dataloader)
        validation_loss_list += [validation_loss]

        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")
        print()

    return train_loss_list, validation_loss_list

train_loss_list, validation_loss_list = fit(model, opt, loss_fn, train_dataloader, val_dataloader, 10)




plt.plot(train_loss_list, label = "Train loss")
plt.plot(validation_loss_list, label = "Validation loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.legend()
plt.show()

def predict(model, input_sequence, max_length=15, SOS_token=2, EOS_token=3):
    model.eval()

    y_input = torch.tensor([[SOS_token]], dtype=torch.long, device=device)

    num_tokens = len(input_sequence[0])

    for _ in range(max_length):
        # Get source mask
        tgt_mask = model.get_tgt_mask(y_input.size(1)).to(device)

        pred = model(input_sequence, y_input, tgt_mask)

        next_item = pred.topk(1)[1].view(-1)[-1].item() # num with highest probability
        next_item = torch.tensor([[next_item]], device=device)

        # Concatenate previous input with predicted best word
        y_input = torch.cat((y_input, next_item), dim=1)

        # Stop if model predicts end of sentence
        if next_item.view(-1).item() == EOS_token:
            break

    return y_input.view(-1).tolist()


# Here we test some examples to observe how the model predicts
examples = [
    torch.tensor([[2, 0, 0, 0, 0, 0, 0, 0, 0, 3]], dtype=torch.long, device=device),
    torch.tensor([[2, 1, 1, 1, 1, 1, 1, 1, 1, 3]], dtype=torch.long, device=device),
    torch.tensor([[2, 1, 0, 1, 0, 1, 0, 1, 0, 3]], dtype=torch.long, device=device),
    torch.tensor([[2, 0, 1, 0, 1, 0, 1, 0, 1, 3]], dtype=torch.long, device=device),
    torch.tensor([[2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 3]], dtype=torch.long, device=device),
    torch.tensor([[2, 0, 1, 3]], dtype=torch.long, device=device)
]

for idx, example in enumerate(examples):
    result = predict(model, example)
    print(f"Example {idx}")
    print(f"Input: {example.view(-1).tolist()[1:-1]}")
    print(f"Continuation: {result[1:-1]}")
    print()