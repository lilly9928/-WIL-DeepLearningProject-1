import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import torchvision.utils
from torchvision import transforms
from torch.utils.data import DataLoader
from imagedata import ImageData
from torch.optim import lr_scheduler
from stnNetwork import Network, init_weights
from trainer import fit
from tripletloss import TripletLoss
from torch.utils.tensorboard import SummaryWriter
from xgboost import XGBClassifier
import time
import copy



#hyperparmeters
batch_size = 1000
epochs =50
learning_rate=0.001
embedding_dims = 2

#data_aug
train_df = pd.read_csv("C:/Users/1315/Desktop/clean/data/ck_train.csv")
test_df = pd.read_csv("C:/Users/1315/Desktop/clean/data/ck_val.csv")

transformation = transforms.Compose([transforms.ToTensor(),])

#data_aug
train_df = pd.read_csv("C:/Users/1315/Desktop/clean/data/ck_train.csv")
test_df = pd.read_csv("C:/Users/1315/Desktop/clean/data/ck_val.csv")

color_jitter = transforms.ColorJitter(0.8 * 1, 0.8 * 1, 0.8 * 1, 0.2 * 1)

train_ds = ImageData(train_df,
                 train=True,
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     #transforms.RandomApply([color_jitter], p=0.8),
                 ]))
test_ds = ImageData(test_df, train=True, transform=transforms.Compose([
                     transforms.ToTensor()
                 ]))

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Network().to(device)
criterion =nn.TripletMarginLoss(margin=1.0, p=2)
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)


fit(train_loader, test_loader, model, criterion, optimizer, scheduler, epochs, device, 50)

