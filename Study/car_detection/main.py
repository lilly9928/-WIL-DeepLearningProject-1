import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import cv2
from torch import optim
from torch.utils.data import DataLoader
from torch import nn
from torchvision.transforms import Compose, ToTensor
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision.models as models
from dataloader import Dataset
import numpy as np
import torch.nn.functional as F

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = models.resnet18()
# model.fc = nn.Linear(512,2)
# model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=3, bias=False)
# model.to(device)
class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.model = models.resnet18()
        self.model.fc = nn.Linear(512, 2)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=3, bias=False)

    def forward(self, x):
        x = self.model(x)
        return x

# 모델 초기화
model = TheModelClass()
model.to(device)


epochs=50

transformation = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(p=0.2),
    ])



dataset = Dataset('D:/data/car_data/', 'D:/data/car_data/info.txt',transform=transformation)

train_loader = torch.utils.data.DataLoader(
    dataset     = dataset,
    batch_size  = 10,
    shuffle     = True,
    num_workers = 0
)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion =nn.CrossEntropyLoss()
running_loss = []
train_correct=0

for epoch in range(epochs):
    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(data)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss.append(loss.cpu().detach().numpy())
        _, preds = torch.max(outputs,1)
        train_correct = torch.sum(preds == labels.data)
        train_acc = train_correct.double() / len(labels)

        print("Epoch: {}/{} - Loss: {:.4f} Training Acuuarcy {:.3f}% ".format(epoch + 1, epochs, np.mean(running_loss),
                                                                              train_acc * 100))




PATH = './car.pth'
torch.save(model.state_dict(), PATH)

