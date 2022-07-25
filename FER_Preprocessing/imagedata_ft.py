
import torch
import numpy as np
from PIL import Image
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import cv2
from skimage import exposure
from skimage.feature import hog
import torchvision.transforms as transforms


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
        src =cv2.imread(self.img_dir + self.datatype + str(idx) + '.jpg',0)
        img= cv2.resize(src,(244,244))
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift))
        lables = np.array(self.lables[idx])
        lables = torch.from_numpy(lables).long()

        if self.transform:
            img = self.transform(magnitude_spectrum)

        return img,lables