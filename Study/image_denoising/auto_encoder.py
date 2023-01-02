
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torchvision
import torch.nn.functional as F
from torch import nn, optim
from torchvision import transforms, datasets

import matplotlib.pyplot as plt
import numpy as np
from dataloader import FlickerDataset,Dataset
from PIL import Image
import cv2
import math
import random

def add_noise(img):
    # Getting the dimensions of the image
    row, col = img.shape

    # Randomly pick some pixels in the
    # image for coloring them white
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)

        # Color that pixel to white
        img[y_coord][x_coord] = 255

    # Randomly pick some pixels in
    # the image for coloring them black
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)

        # Color that pixel to black
        img[y_coord][x_coord] = 0

    return img
# 하이퍼파라미터
EPOCH = 100
BATCH_SIZE = 8
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    ])

noise_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    ])


dataset = FlickerDataset('D:/data/vqa/flickr8k/images/','D:/data/vqa/flickr8k/denoise.txt', transform=transform,transform_noise=noise_transform)
dataset_1 = Dataset('D:/data/test/', 'D:/data/test/info.txt',transform=transform,transform_noise=noise_transform)

train_loader = torch.utils.data.DataLoader(
    dataset     = dataset_1,
    batch_size  = 1,
    shuffle     = True,
    num_workers = 0
)



class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(256*256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3),   # 입력의 특징을 3차원으로 압축합니다
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256*256),
            nn.Sigmoid(),       # 픽셀당 0과 1 사이로 값을 출력합니다
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

autoencoder = Autoencoder().to(DEVICE)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.005)
criterion = nn.MSELoss()


def train(autoencoder, train_loader):
    autoencoder.train()
    avg_loss = 0
    for step, (x,noise_x) in enumerate(train_loader):
        noisy_x = noise_x
        noisy_x = noisy_x.view(-1, 256 * 256).to(DEVICE)
        y = x.view(-1, 256 * 256).to(DEVICE)

        encoded, decoded = autoencoder(noisy_x)

        loss = criterion(decoded, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()
    return avg_loss / len(train_loader)

# for epoch in range(1, EPOCH+1):
#     loss = train(autoencoder, train_loader)
#     print("[Epoch {}] loss:{}".format(epoch, loss))
#
# autoencoder.eval()
# gray_img= cv2.imread(os.path.join("D:/data/test2.bmp"),cv2.IMREAD_GRAYSCALE)
# plt.imshow(gray_img, cmap='gray')
# plt.savefig('orgin_2.jpg')
# plt.show()
# # _,res_1 = autoencoder(transform(img1).view(-1, 256 * 256).to(DEVICE))
# _,res_2 = autoencoder(noise_transform(gray_img).view(-1, 256 * 256).to(DEVICE))
#
# res_2 = np.reshape(res_2.to("cpu").data.numpy(), (256, 256,-1))
# plt.imshow(res_2, cmap='gray')
# plt.savefig('result2.jpg')
src = cv2.imread("D:/data/lena.jpg",cv2.IMREAD_GRAYSCALE)
src = cv2.resize(src,(512,512))
contrast = cv2.imread("result.jpg",cv2.IMREAD_GRAYSCALE)
contrast = cv2.resize(contrast,(512,512))

src1 = cv2.imread("D:/data/fce5.jpg",cv2.IMREAD_GRAYSCALE)
src1 = cv2.resize(src,(512,512))
contrast1 = cv2.imread("result2.jpg",cv2.IMREAD_GRAYSCALE)
contrast1 = cv2.resize(contrast,(512,512))

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

d = psnr(src, contrast)
print(d)
d1 = psnr(src1, contrast1)
print(d1)

# plt.imshow(src)
# plt.imshow(contrast)

plt.show()