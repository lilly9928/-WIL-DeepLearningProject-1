import torch
import pandas as pd
import random
import cv2
import numpy as np
from torchvision import transforms
from torch.utils.data import  Dataset
from skimage import exposure
from skimage.feature import hog

class ImageData(Dataset):

    def __init__(self, csv_file, img_dir, datatype, transform):


        self.csv_file = pd.read_csv(csv_file)
        self.lables = self.csv_file['emotion']
        self.img_dir = img_dir
        self.transform = transform
        self.datatype = datatype
        self.index = self.csv_file.index.values



    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        src = cv2.imread(self.img_dir + self.datatype + str(idx) + '.jpg')
        img = cv2.resize(src, (224, 224))
        lables = np.array(self.lables[idx])
        lables = torch.from_numpy(lables).long()

        if self.datatype == 'ck_train': #train 데이터 일 경우

            ## 해당 anchor가 아닌 것들중에서 Label 같은 것들의 index를 가지고 옮
            positive_list = self.index[self.index != idx][self.lables[self.index != idx] == int(lables)]

            positive_item = random.choice(positive_list)
            positive_src = cv2.imread(self.img_dir + self.datatype + str(positive_item) + '.jpg')
            positive_img = cv2.resize(positive_src, (224, 224))

            ## 해당 anchor가 아닌 것들중에서 Label 다른 것들의 index를 가지고 옮
            negative_list = self.index[self.index != idx][self.lables[self.index != idx] != int(lables)]

            nagative_item = random.choice(negative_list)
            negative_src = cv2.imread(self.img_dir + self.datatype + str(nagative_item) + '.jpg')
            negative_img = cv2.resize(negative_src, (224, 224))

            if self.transform:
                anchor_img = self.transform(img)
                positive_img = self.transform(positive_img)
                negative_img = self.transform(negative_img)

            return anchor_img, positive_img, negative_img, lables

        else:
            if self.transform: #val 데이터 일 경우
                anchor_img = self.transform(img)

            return anchor_img,lables