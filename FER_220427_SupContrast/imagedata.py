
import torch
import numpy as np
from PIL import Image
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset

class ImageData(Dataset):

    def __init__(self, csv_file, img_dir, datatype, transform):
        '''
        Pytorch Dataset class
        params:-
                 csv_file : the path of the csv file    (train, validation, test)
                 img_dir  : the directory of the images (train, validation, test)
                 datatype : string for searching along the image_dir (train, val, test)
                 transform: pytorch transformation over the data_aug
        return :-
                 image, labels
        '''
        self.csv_file = pd.read_csv(csv_file)
        self.lables = self.csv_file['emotion']
        self.img_dir = img_dir
        self.transform = transform
        self.datatype = datatype

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = Image.open(self.img_dir + self.datatype + str(idx) + '.jpg').convert('RGB')
        lables = np.array(self.lables[idx])
        lables = torch.from_numpy(lables).long()

        if self.transform:
            img = self.transform(img)

        return img, lables