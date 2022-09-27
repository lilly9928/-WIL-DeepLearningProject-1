import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

from raf_imagedata import RafDataset
#from imagedata import ImageData
from imagedata_ft import ImageData

from network import Resnet
from utils import train_val
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 64

#image load
# train_csvdir = 'D:/data/FER/ck_images/ck_train.csv'
# traindir = "D:/data/FER/ck_images/Images/ck_train/"
# val_csvdir= 'D:/data/FER/ck_images/ck_val.csv'
# valdir = "D:/data/FER/ck_images/Images/ck_val/"
#
# transformation = transforms.Compose([transforms.ToTensor()])
# train_dataset =ImageData(csv_file = train_csvdir, img_dir = traindir, datatype = 'ck_train',transform = transformation)
# train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
#
# val_dataset =ImageData(csv_file = val_csvdir, img_dir = valdir, datatype = 'ck_val',transform = transformation)
# val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    #transforms.Normalize((0.5611489, 0.44190985, 0.39697975), (0.21449453, 0.19619425, 0.18772252))
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
    #transforms.RandomErasing(scale=(0.02, 0.25))
    ])

eval_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])

hog_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
    ])

train_dataset= RafDataset(path='D:\\data\\FER\\RAF\\basic', phase='train', transform=eval_transforms)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

val_dataset= RafDataset(path='D:\\data\\FER\\RAF\\basic', phase='test', transform=eval_transforms)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

model = Resnet(base_model="resnet18",out_dim=7).to(device)

loss_func = nn.CrossEntropyLoss(reduction='sum')
opt = optim.Adam(model.parameters(), lr=0.01)
lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=10)

# classes = ['Happily Surprised', 'Happily Disgusted', 'Sadly Fearful', 'Sadly Angry', 'Sadly Surprised',
#              'Sadly Disgusted', 'Fearfully Angry',
#              'Fearfully Surprised', 'Angrily Surprised', 'Angrily Disgusted', 'Disgustedly Surprised']

#mean_, std_ = calculate_norm(train_dataset)
#print(f'평균(R,G,B): {mean_}\n표준편차(R,G,B): {std_}')


# define the training parameters
params_train = {
    'num_epochs':100,
    'optimizer':opt,
    'loss_func':loss_func,
    'train_dl':train_loader,
    'val_dl':val_loader,
    'sanity_check':False,
    'lr_scheduler':lr_scheduler,
    'path2weights':'./models/weights.pt',
}

model, loss_hist, metric_hist = train_val(model,params_train)
#
# # train-val progress
# num_epochs = params_train['num_epochs']
#
# # plot loss progress
# plt.title('Train-Val Loss')
# plt.plot(range(1, num_epochs+1), loss_hist['train'], label='train')
# plt.plot(range(1, num_epochs+1), loss_hist['val'], label='val')
# plt.ylabel('Loss')
# plt.xlabel('Training Epochs')
# plt.legend()
# plt.show()
#
# # plot accuracy progress
# plt.title('Train-Val Accuracy')
# plt.plot(range(1, num_epochs+1), metric_hist['train'], label='train')
# plt.plot(range(1, num_epochs+1), metric_hist['val'], label='val')
# plt.ylabel('Accuracy')
# plt.xlabel('Training Epochs')
# plt.legend()
# plt.show()