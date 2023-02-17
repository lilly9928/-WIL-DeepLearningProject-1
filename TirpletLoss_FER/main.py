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
from transformNetwork import Network, init_weights,ResNetMultiStn,Bottleneck
from torch.utils.tensorboard import SummaryWriter
from xgboost import XGBClassifier
from raf_imagedata import RafDataset


torch.cuda.empty_cache()

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
learning_rate=0.001
embedding_dims = 2

#data_aug
train_csvdir = 'D:/data/FER/ck_images/ck_train.csv'
traindir = "D:/data/FER/ck_images/Images/ck_train/"
val_csvdir= 'D:/data/FER/ck_images/ck_val.csv'
valdir = "D:/data/FER/ck_images/Images/ck_val/"

eval_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])

transformation = transforms.Compose([transforms.ToTensor()])
train_dataset =ImageData(csv_file = train_csvdir, img_dir = traindir, datatype = 'ck_train',transform = transformation)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

val_dataset =ImageData(csv_file = val_csvdir, img_dir = valdir, datatype = 'ck_val',transform = transformation)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)


# train_dataset= RafDataset(path='D:\\data\\FER\\RAF\\basic', phase='train', transform=eval_transforms)
# train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
#
# val_dataset= RafDataset(path='D:\\data\\FER\\RAF\\basic', phase='test', transform=eval_transforms)
# val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

#device, model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNetMultiStn(Bottleneck, [3, 4, 6, 3]).to(device)
#model = Network().to(device)
model.apply(init_weights)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
Triplet_criterion =nn.TripletMarginLoss(margin=1.0, p=2)
criterion = nn.CrossEntropyLoss(reduction='sum')

model.train()

writer = SummaryWriter(f'runs/ck/MiniBatchsize {batch_size} LR {learning_rate}_220905')
#writer = SummaryWriter(f'runs/FER/image_test')
# ck+ class
classes = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

# classes = ['Surprise','Fear','Disgust','Happiness','Sadness','Anger','Neutral']


for epoch in range(epochs):
    running_loss = []
    accuracies = []
    train_correct = 0

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

        features = anchor_img.reshape(anchor_img.shape[0], -1)
        _, preds = torch.max(anchor_out, 1)
        train_correct += torch.sum(preds == anchor_label.data)


        #writer.add_embedding(features, metadata=class_labels, label_img=anchor_img,global_step=batch_idx)

    train_acc = train_correct.double() / len(train_dataset)
    print(f'Mean Loss this epoch was {sum(running_loss) / len(running_loss)}')
    writer.add_scalar('Training Loss',sum(running_loss) / len(running_loss), global_step=epoch)
   # print("Epoch: {}/{} - Loss: {:.4f}".format(epoch + 1, epochs, np.mean(running_loss)))
    print("Epoch: {}/{} - Loss: {:.4f} Training Acuuarcy {:.3f}% ".format(epoch + 1, epochs, np.mean(running_loss),train_acc * 100))

torch.save(model.state_dict(),'train01.pt')
#
# #check_accarcy
model.eval()

num_correct = 0
num_sample = 0

train_results = []
labels = []

model.eval()
model.load_state_dict(torch.load('train01.pt'))

def show(img, y=None):
    npimg = img.numpy()
    npimg_tr = np.transpose(npimg, (1, 2, 0))
    plt.imshow(npimg_tr)

    if y is not None:
        plt.title('labels:' + str(y))

with torch.no_grad():
    for img,label in val_loader:
        img = img.to(device)
        gt = label
        _,out = model(img)
        _, predicted = torch.max(out, 1)

        x_grid = img

        break

x_grid = torchvision.utils.make_grid(x_grid.cpu(), nrow=8, padding=2)
show(x_grid)
plt.show()
print('GT: ', ''.join(' %s' % classes[label[j]]for j in range(8)))
print('Predicted: ', ''.join(' %s' % classes[predicted[j]]for j in range(8)))

# imshow(torchvision.utils.make_grid(images))
# print('GroundTruth: ', ''.join(' %s' % classes[labels[j]] for j in range(8)))
#
# model.load_state_dict(torch.load('train01.pt'))
# _,outputs = model(images)
# _, predicted = torch.max(outputs, 1)
#
# print('Predicted: ', ''.join(' %s' % classes[predicted[j]]for j in range(8)))


# with torch.no_grad():
#     for img, _, _, label in train_loader:
#         img, label = img.to(device), label.to(device)
#         train_results.append(model(img))
#         labels.append(label.cpu())
#
# train_results = np.concatenate(train_results)
# labels = np.concatenate(labels)
# labels.shape
#
# ## visualization
# plt.figure(figsize=(15, 10), facecolor="azure")
# for label in np.unique(labels):
#     tmp = train_results[labels == label]
#     plt.scatter(tmp[:, 0], tmp[:, 1], label=classes[label])
#
# plt.legend()
# plt.show()
#
# tree = XGBClassifier(seed=180)
# tree.fit(train_results, labels)
#
# test_results = []
# test_labels = []
#
# model.eval()
# with torch.no_grad():
#     for img,label in val_loader:
#         img = img.to(device)
#         test_results.append(model(img).cpu().numpy())
#         test_labels.append(tree.predict(model(img).cpu().numpy()))
#
# test_results = np.concatenate(test_results)
# test_labels = np.concatenate(test_labels)
#
# plt.figure(figsize=(15, 10), facecolor="azure")
# for label in np.unique(test_labels):
#     tmp = test_results[test_labels == label]
#     plt.scatter(tmp[:, 0], tmp[:, 1], label=classes[label])
#
# plt.legend()
# plt.show()
#
# # accuracy
# true_ = (tree.predict(test_results) == test_labels).sum()
# len_ = len(test_labels)
# print("Accuracy :{}%".format((true_ / len_) * 100))  ##100%

# dataiter = iter(test_loader)
# x0,_ = next(dataiter)
#
# for i in range(10):
#     x1, label2 = next(dataiter)
#     concatenated = torch.cat((x0, x1), 0)
#
#     output1, output2 = model(Variable(x0).cuda(), Variable(x1).cuda())
#     euclidean_distance = F.pairwise_distance(output1, output2)
#     imshow(torchvision.utils.make_grid(concatenated), 'Dissimilarity: {:.2f}'.format(euclidean_distance.item()))
