import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import random
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.utils
from torchvision import transforms
from torch.utils.data import DataLoader
from imagedata import ImageData
from transformNetwork448 import init_weights,Network,Bottleneck
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from fer_imagedata import FERimageData
from raf_imagedata import RafDataset
from sklearn import metrics
from torch.nn import functional as F
import skimage.transform


torch.cuda.empty_cache()

import sys


def printsave(*a):
    file = open('D:\\GitHub\\-WIL-Expression-Recognition-Study\\TirpletLoss_FER', 'a')
    print(*a)
    print(*a, file=file)


def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# torch.manual_seed(2020)
# np.random.seed(2020)
# random.seed(2020)


#hyperparmeters
batch_size = 8
epochs =100
learning_rate=0.01
embedding_dims = 2
num_layers = 3
parallels=[0.5]

train_csvdir = 'D:/data/FER/ck_images/ck_train.csv'
traindir = "D:/data/FER/ck_images/Images/ck_train/"
val_csvdir= 'D:/data/FER/ck_images/ck_val.csv'
valdir = "D:/data/FER/ck_images/Images/ck_val/"

fer_train_csvdir = 'D:/data/FER/train.csv'
fer_traindir = "D:/data/FER/train/"
fer_val_csvdir = 'D:/data/FER/train_val.csv'
fer_test_csvdir= 'D:/data/FER/val_1.csv'
fer_testdir = "D:/data/FER/val_1/"

eval_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((448, 448)),
    transforms.ToTensor()

    ])

transformation = transforms.Compose([transforms.ToTensor()])
writer = SummaryWriter(f'runs/raf/MiniBatchsize {batch_size} LR {learning_rate}')
#fer
# train_dataset =FERimageData(csv_file = fer_train_csvdir, img_dir = fer_traindir, datatype = 'train',transform = transformation)
# train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
#
# val_dataset =FERimageData(csv_file = fer_val_csvdir, img_dir = fer_traindir, datatype = 'train',transform = transformation)
# val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
#
# test_dataset =FERimageData(csv_file = fer_test_csvdir, img_dir = fer_testdir, datatype = 'val_1',transform = transformation)
# test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

#ck+
# train_dataset =ImageData(csv_file = train_csvdir, img_dir = traindir, datatype = 'ck_train',transform = transformation)
# train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
#
# val_dataset =ImageData(csv_file = val_csvdir, img_dir = valdir, datatype = 'ck_val',transform = transformation)
# val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=True)

#raf
train_dataset= RafDataset(path='D:/data/FER/RAF/basic', phase='train', transform=eval_transforms)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# val_dataset= RafDataset(path='D:/data/FER/RAF/basic', phase='test', transform=eval_transforms)
# val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)

#device, model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model = Network(Bottleneck, [3, 4, 6, 3],num_layers=3).to(device)
# model.apply(init_weights)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Triplet_criterion =nn.TripletMarginLoss(margin=0.5, p=2)
# criterion = nn.CrossEntropyLoss(reduction='sum')
#
# model.train()

# ck+ class
#classes = ['AN','DI','FE','HA','SA','SU','NE']

classes = ['SU','FE','DI','HA','SA','AN','NE']

for scale in parallels:

    average_acc = 0

    parallel = [scale,scale,scale]
    model = Network(Bottleneck, [3, 4, 6, 3],parallel=parallel, num_layers=3).to(device)
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    Triplet_criterion = nn.TripletMarginLoss(margin=0.5, p=2)
    criterion = nn.CrossEntropyLoss(reduction='sum')

    model.train()

    for epoch in range(epochs):
        running_loss = []
        accuracies = []
        train_correct = 0
        print("epoch:",epoch,"start")
        for batch_idx, (anchor_img, positive_img, negative_img, anchor_label) in enumerate(train_loader):

            #get data_aug to cuda
            anchor_img, positive_img, negative_img =\
                anchor_img.to(device), positive_img.to(device), negative_img.to(device)

            anchor_label = anchor_label.to(device)

            optimizer.zero_grad()

            anchor_feature,anchor_out = model(anchor_img)
            positive_feature,positive_out = model(positive_img)
            negative_feature,negative_out = model(negative_img)


            Triplet_loss = Triplet_criterion(anchor_feature, positive_feature, negative_feature)
            entropy_loss=criterion(anchor_out,anchor_label)

            loss = Triplet_loss+entropy_loss

            loss.backward()

            optimizer.step()

            running_loss.append(loss.cpu().detach().numpy())
            class_labels = [classes[label] for label in anchor_label]

            features = anchor_feature.reshape(anchor_img.shape[0], -1)
            _, preds = torch.max(anchor_out, 1)
            train_correct += torch.sum(preds == anchor_label.data)


        writer.add_scalar('Training Loss', loss)

        writer.add_embedding(features, metadata=class_labels, label_img=anchor_img,global_step=None)

        train_acc = train_correct.double() / len(train_dataset)
        average_acc += train_acc
        print(f'Mean Loss this epoch was {sum(running_loss) / len(running_loss)}')
        print("Epoch: {}/{} - Loss: {:.4f} Training Accuarcy {:.3f}%".format(epoch + 1, epochs, np.mean(running_loss),train_acc * 100))

    average_acc = average_acc.cpu()/epochs
    torch.save(model.state_dict(),'train_raf_scale_'+str(scale)+'_acc_'+str(average_acc)+'.pt')


# #check_accarcy
#model.eval()

num_correct = 0
num_sample = 0

train_results = []
labels = []

# model.eval()
# model.load_state_dict(torch.load('train_raf_448.pt'))
#
# def show(img, y=None):
#     npimg = img.numpy()
#     npimg_tr = np.transpose(npimg, (1, 2, 0))
#     plt.imshow(npimg_tr)
#
#     if y is not None:
#         plt.title('labels:' + str(y))
#
# class SaveFeatures():
#     features=None
#     def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
#     def hook_fn(self, module, input, output): self.features = ((output.cpu()).data).numpy()
#     def remove(self): self.hook.remove()
#
#
# def getCAM(feature_conv, weight_fc):
#     _, nc, h, w = feature_conv.shape
#     cam = weight_fc[0].dot(feature_conv.reshape((nc, h*w)))
#     cam = cam.reshape(h, w)
#     cam = cam - np.min(cam)
#     cam_img = cam / np.max(cam)
#     return [cam_img]


# look_src = []
# confusion_label=[]
# confusion_predicted=[]
#
# display_transform = transforms.Compose([
#    transforms.Resize((224,224))])
#
# flag = 0
# #test datan
# with torch.no_grad():
#
#     correct = 0
#     for img,_,_,label in train_loader:
#
#         #look_src += src
#         img = img.to(device)
#         label = label.to(device)
#         prediction_var = Variable((img.unsqueeze(0)).cuda(), requires_grad=True)
#
#         final_layer = model._modules.get('layer4')
#         activated_features = SaveFeatures(final_layer)
#
#         feature,out = model(img)
#         _, predicted = torch.max(out, 1)
#
#         correct += torch.sum(predicted == label.data)
#
#         confusion_label+= label.cpu()
#         confusion_predicted+=predicted.cpu()
#         x_grid = img
#
#     acc = correct.double() / len(train_dataset)
#
# confusion_matrix = metrics.confusion_matrix(confusion_label, confusion_predicted)
#
# confusion_matrix_norm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
# cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_norm,display_labels=classes)
#
# cm_display.plot(cmap=plt.cm.Blues)
# plt.show()
#
# print('GT: ', ''.join(' %s' % classes[confusion_label[j]]for j in range(8)))
# print('Predicted: ', ''.join(' %s' % classes[confusion_predicted[j]]for j in range(8)))
# print('acc',acc)
