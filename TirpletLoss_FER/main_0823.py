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
from torchsummary import summary
from transformNetwork import Network, init_weights
from tripletloss import TripletLoss
from torch.utils.tensorboard import SummaryWriter
from woResnet import Emotion
from xgboost import XGBClassifier
import time
import copy



#hyperparmeters
batch_size = 1000
epochs =50
learning_rate=0.001
embedding_dims = 2

#data_aug
train_df = pd.read_csv("C:/Users/1315/Desktop/data/ck_train.csv")
test_df = pd.read_csv("C:/Users/1315/Desktop/data/ck_val.csv")

transformation = transforms.Compose([transforms.ToTensor(),])

train_ds = ImageData(train_df, train=True, transform=transformation)

test_ds = ImageData(test_df, train=False, transform=transformation)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Network().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion =nn.TripletMarginLoss(margin=1.0, p=2)

