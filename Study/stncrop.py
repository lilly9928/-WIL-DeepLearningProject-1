from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import torch.optim as optim
from raf import RafDataset
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


img = cv2.imread('C:/Users/1315/Desktop/test/happy_man.jpg',0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64

eval_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.Grayscale(1)
]
)


train_dataset= RafDataset(path='D:\\data\\FER\\RAF\\basic', phase='train', transform=eval_transforms)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

val_dataset= RafDataset(path='D:\\data\\FER\\RAF\\basic', phase='test', transform=eval_transforms)
test_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # 공간 변환을 위한 위치 결정 네트워크 (localization-network)
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # [3 * 2] 크기의 아핀(affine) 행렬에 대해 예측
        self.fc_loc = nn.Sequential(
            nn.Linear(10*3*3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # 항등 변환(identity transformation)으로 가중치/바이어스 초기화
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # STN의 forward 함수
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10*3*3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # 입력을 변환
        x = self.stn(x)

        # 일반적인 forward pass를 수행
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = Net().to(device)

optimizer = optim.SGD(model.parameters(), lr=0.01)


def train(epoch):
    model.train()
    for batch_idx, (_,data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
#
# MNIST 데이터셋에서 STN의 성능을 측정하기 위한 간단한 테스트 절차
#


def test():
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for _, data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # 배치 손실 합하기
            test_loss += F.nll_loss(output, target, size_average=False).item()
            # 로그-확률의 최대값에 해당하는 인덱스 가져오기
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
              .format(test_loss, correct, len(test_loader.dataset),
                      100. * correct / len(test_loader.dataset)))

def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

# 학습 후 공간 변환 계층의 출력을 시각화하고, 입력 이미지 배치 데이터 및
# STN을 사용해 변환된 배치 데이터를 시각화 합니다.


def visualize_stn():
    with torch.no_grad():
        # 학습 데이터의 배치 가져오기
        data = next(iter(test_loader))[1].to(device)

        input_tensor = data.cpu()
        transformed_input_tensor = model.stn(data).cpu()

        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))

        # 결과를 나란히 표시하기
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')

        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')

for epoch in range(1, 20 + 1):
    train(epoch)
    test()

# 일부 입력 배치 데이터에서 STN 변환 결과를 시각화
visualize_stn()

plt.show()