import os
import pandas as pd
from PIL import Image
import numpy as np
import cv2
import random
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
class Dataset(Dataset):
    def __init__(self, root_dir,file, transform=None):
        self.root_dir = root_dir
        self.df = pd.read_csv(file)
        self.transform = transform
        self.imgs = self.df["image"]
        self.label = self.df["label"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_id = self.imgs[index]
        img_label = self.label[index]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("L")
        if self.transform is not None:
            img = self.transform(img)

        return img,img_label
