import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from imagedata import ImageData
#from imagedata_ft import ImageData
from network import Resnet
from utils import train_val
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 8

#image load
train_csvdir= 'C:/Users/1315/Desktop/data/train.csv'
traindir = "C:/Users/1315/Desktop/data/train/"
val_csvdir= 'C:/Users/1315/Desktop/data/finaltest.csv'
valdir = "C:/Users/1315/Desktop/data/finaltest/"

transformation = transforms.Compose([transforms.ToTensor()])
train_dataset =ImageData(csv_file = train_csvdir, img_dir = traindir, datatype = 'train',transform = transformation)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

val_dataset =ImageData(csv_file = val_csvdir, img_dir = valdir, datatype = 'finaltest',transform = transformation)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

model = Resnet(base_model="resnet18",out_dim=7).to(device)

loss_func = nn.CrossEntropyLoss(reduction='sum')
opt = optim.Adam(model.parameters(), lr=0.01)
lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=10)


classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
#
# #check sample images
# def show(img, y=None):
#     npimg = img.numpy()
#     npimg_tr = np.transpose(npimg, (1, 2, 0))
#     plt.imshow(npimg_tr)
#
#     if y is not None:
#         plt.title('labels:' + str(y))
#
#
# grid_size=4
# rnd_ind = np.random.randint(0, len(train_dataset), grid_size)
#
# x_grid = [train_dataset[i][0] for i in rnd_ind]
# y_grid = [classes[train_dataset[i][1]] for i in rnd_ind]
#
# x_grid = torchvision.utils.make_grid(x_grid, nrow=grid_size, padding=2)
# plt.figure(figsize=(10,10))
# show(x_grid, y_grid)
# plt.show()


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

# train-val progress
num_epochs = params_train['num_epochs']

# plot loss progress
plt.title('Train-Val Loss')
plt.plot(range(1, num_epochs+1), loss_hist['train'], label='train')
plt.plot(range(1, num_epochs+1), loss_hist['val'], label='val')
plt.ylabel('Loss')
plt.xlabel('Training Epochs')
plt.legend()
plt.show()

# plot accuracy progress
plt.title('Train-Val Accuracy')
plt.plot(range(1, num_epochs+1), metric_hist['train'], label='train')
plt.plot(range(1, num_epochs+1), metric_hist['val'], label='val')
plt.ylabel('Accuracy')
plt.xlabel('Training Epochs')
plt.legend()
plt.show()