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
from imagedata_3 import ImageData
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from xgboost import XGBClassifier
from network import Network



# torch.manual_seed(2020)
# np.random.seed(2020)
# random.seed(2020)


#hyperparmeters
batch_size = 8
epochs =10
learning_rate=0.1

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


writer = SummaryWriter(f'runs/MiniBatchsize {batch_size} LR {learning_rate}_220905')
#writer = SummaryWriter(f'runs/FER/image_test')
classes = ['0','1','2','3','4','5','6']

#device, model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Network().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion =nn.TripletMarginLoss(margin=1.0, p=2)

model.train()

for epoch in range(epochs):
    running_loss = []
    accuracies = []

    for batch_idx, (anchor_img, positive_img, negative_img, anchor_label) in enumerate(train_loader):

        #get data_aug to cuda
        anchor_img, positive_img, negative_img =\
            anchor_img.to(device), positive_img.to(device), negative_img.to(device)

        optimizer.zero_grad()
        anchor_out,positive_out,negative_out = model(anchor_img,positive_img,negative_img)

        img_grid = torchvision.utils.make_grid(anchor_img)
        img_grid1 = torchvision.utils.make_grid(positive_img)
        img_grid2 = torchvision.utils.make_grid(negative_img)
      #  image_grid3 = torchvision.utils.make_grid(model.stn(anchor_img))


        loss = criterion(anchor_out, positive_out, negative_out)
        loss.backward()
        optimizer.step()

        running_loss.append(loss.cpu().detach().numpy())

        features = anchor_img.reshape(anchor_img.shape[0], -1)
        class_labels = [classes[label] for label in anchor_label]

        writer.add_image('anchor', img_grid)
        writer.add_image('positive', img_grid1)
        writer.add_image('negative', img_grid2)
     #   writer.add_image('layer', image_grid3)

        #writer.add_embedding(features, metadata=class_labels, label_img=anchor_img,global_step=batch_idx)

    print(f'Mean Loss this epoch was {sum(running_loss) / len(running_loss)}')
    writer.add_scalar('Training Loss',sum(running_loss) / len(running_loss), global_step=epoch)
    print("Epoch: {}/{} - Loss: {:.4f}".format(epoch + 1, epochs, np.mean(running_loss)))

torch.save(model.state_dict(),'train02.pt')

#check_accarcy
model.eval()

num_correct = 0
num_sample = 0

# with torch.no_grad():
#     for img,label in test_loader:
#         img,label = img.to(device),label.to(device)
#
#         scores = model(img)
#         _,predictions = scores.max(1)
#         num_correct += (predictions == label).sum()
#         num_sample += predictions.size(0)
#
#     print(f"got{num_correct}/{num_sample} with accuarcy {float(num_correct) / float(num_sample) * 100:.2f}")


# train_results = []
# labels = []
# classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
#
# model.eval()
# with torch.no_grad():
#     for img, _, _, label in train_loader:
#         img, label = img.to(device), label.to(device)
#         train_results.append(model(img).cpu().numpy())
#         labels.append(label.cpu())
#
# train_results = np.concatenate(train_results)
# labels = np.concatenate(labels)
# labels.shape

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
#     for img,label in test_loader:
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

dataiter = iter(test_loader)
x0,_,_,_ = next(dataiter)

from torch.autograd import Variable
import torch.nn.functional as F

def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


for i in range(10):
    _,x1,x2, label2 = next(dataiter)
    concatenated = torch.cat((x0, x1), 0)

    output1, output2,output3 = model(Variable(x0).cuda(), Variable(x1).cuda(),Variable(x2).cuda())
    euclidean_distance = F.pairwise_distance(output1, output2)
    imshow(torchvision.utils.make_grid(concatenated), 'Dissimilarity: {:.2f}'.format(euclidean_distance.item()))