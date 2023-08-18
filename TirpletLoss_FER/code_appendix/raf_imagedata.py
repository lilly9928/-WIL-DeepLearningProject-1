# -*- coding: utf-8 -*-
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import os
import cv2
import torch.utils.data as data
import torch
import pandas as pd
import random
import numpy as np



def generate_flip_grid(w, h, device):
    # used to flip attention maps
    x_ = torch.arange(w).view(1, -1).expand(h, -1)
    y_ = torch.arange(h).view(-1, 1).expand(-1, w)
    grid = torch.stack([x_, y_], dim=0).float().to(device)
    grid = grid.unsqueeze(0).expand(1, -1, -1, -1)
    grid[:, 0, :, :] = 2 * grid[:, 0, :, :] / (w - 1) - 1
    grid[:, 1, :, :] = 2 * grid[:, 1, :, :] / (h - 1) - 1
    grid[:, 0, :, :] = -grid[:, 0, :, :]
    return grid

def flip_image(image_array):
    return cv2.flip(image_array, 1)

def add_g(image_array, mean=0.0, var=30):
    std = var ** 0.5
    image_add = image_array + np.random.normal(mean, std, image_array.shape)
    image_add = np.clip(image_add, 0, 255).astype(np.uint8)
    return image_add

class RafDataset(data.Dataset):
    def __init__(self, path, phase, basic_aug=True, transform=None):
        self.path =path
        self.phase = phase
        self.basic_aug = basic_aug
        self.transform = transform
        df = pd.read_csv(os.path.join(self.path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None)

        name_c = 0 # 0 인덱스 (파일명)
        label_c = 1 # 1 인덱스 (라벨)
        if phase == 'train':
            dataset = df[df[name_c].str.startswith('train')] #0번째 인덱스(이름)에서 train으로 시작하는 파일
            self.index =dataset.index.values
        else:
            df = pd.read_csv(os.path.join(self.path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None)
            dataset = df[df[name_c].str.startswith('test')] #0번째 인덱스(이름)에서 test으로 시작하는 파일
            self.index = dataset.index.values

        self.label = dataset.iloc[:, label_c].values - 1
        images_names = dataset.iloc[:, name_c].values
        self.aug_func = [flip_image, add_g]
        self.file_paths = []
        # self.clean = ('list_patition_label.txt' == 'list_patition_label.txt')

        for f in images_names:
            f = f.split(".")[0]
            f += '_aligned.jpg'
            file_name = os.path.join(path, "Image/aligned/aligned", f)
            self.file_paths.append(file_name)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        label = self.label[idx]
        image = cv2.imread(self.file_paths[idx])

        image = image[:, :, ::-1]


        if self.phase == 'train':
            positive_list =self.index[self.index!=idx][self.label[self.index!=idx]==int(label)]
            positive_item = random.choice(positive_list)
            positive_image = cv2.imread(self.file_paths[positive_item])

            negative_list=self.index[self.index!=idx][self.label[self.index != idx]!= int(label)]
            negative_item = random.choice(negative_list)
            negative_image = cv2.imread(self.file_paths[negative_item])


            if self.transform is not None:
                image = self.transform(image)
                positive_image = self.transform(positive_image)
                negative_image = self.transform(negative_image)

                return image, positive_image, negative_image, label

        else:

            if self.transform is not None:
                image = self.transform(image)

            return image, self.file_paths,label


        return image, positive_image, negative_image, label
