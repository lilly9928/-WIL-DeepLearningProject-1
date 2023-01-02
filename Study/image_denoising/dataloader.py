import os  # loading file paths
import pandas as pd  # for lookup in annotation file
from torch.utils.data import DataLoader, Dataset
from PIL import Image  # Load img
import torch
import numpy as np
import cv2

import random

# def add_noise(img):
#     noise = torch.randn(img.size()) * 0.2
#     noisy_img = img + noise
#     return noisy_img

def gausian_noise(std, gray):
    gray=cv2.imread(gray)
    gray_img = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    height, width = gray_img.shape
    img_noise = np.zeros((height, width), dtype=np.float)
    for i in range(height):
        for a in range(width):
            make_noise = np.random.normal() # 랜덤함수를 이용하여 노이즈 적용
            set_noise = std * make_noise
            img_noise[i][a] = gray_img[i][a] + set_noise
    return img_noise


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


class FlickerDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None,transform_noise=None):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform
        self.transform_noise = transform_noise
        # img, caption columns
        self.imgs = self.df["image"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("L")
        gray_img= cv2.imread(os.path.join(self.root_dir, img_id),cv2.IMREAD_GRAYSCALE)
        if self.transform is not None:
            img = self.transform(img)
            noise_img = add_noise(gray_img)
            noise_img=self.transform_noise(noise_img.astype('uint8'))

        return img,noise_img


class Dataset(Dataset):
    def __init__(self, root_dir,file, transform=None,transform_noise=None):
        self.root_dir = root_dir
        self.df = pd.read_csv(file)
        self.transform = transform
        self.transform_noise = transform_noise
        self.imgs = self.df["image"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("L")
        gray_img= cv2.imread(os.path.join(self.root_dir, img_id),cv2.IMREAD_GRAYSCALE)
        if self.transform is not None:
            img = self.transform(img)
            noise_img = add_noise(gray_img)
            noise_img=self.transform_noise(noise_img.astype('uint8'))

        return img,noise_img
