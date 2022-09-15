import os
import cv2
import torch.utils.data as data
import torch
import pandas as pd
import random
import numpy as np
from torchvision import transforms

class RafData(data.Dataset):
    def __init__(self,path,phase,basic_aug=True,transform=None):
        self.path = path
        self.phase = phase
        self.basic_aug = basic_aug
        self.transform = transform



if __name__ =="__main__":
    path ='C:\\Users\\1315\\Desktop\\RAF\\compound'
    name_c = 0
    label_c = 1
    df = pd.read_csv(os.path.join(path, 'EmoLabel\\list_patition_label.txt'),sep=' ',header=None)
    print(df)
    dataset = df[df[name_c].str.startswith('train')] #0번째 인덱스(이름)에서 train으로 시작하는 파일
    print(dataset)

    classname=['Happily Surprised','Happily Disgusted','Sadly Fearful','Sadly Angry','Sadly Surprised','Sadly Disgusted','Fearfully Angry',\
               'Fearfully Surprised','Angrily Surprised','Angrily Disgusted','Disgustedly Surprised']
    label = dataset.iloc[:, label_c].values - 1
    images_names = dataset.iloc[:, name_c].values
    print(images_names)

    file_paths = []

    for f in images_names:
        f = f.split(".")[0]
        f += '_aligned.jpg'
        file_name = os.path.join(path, "Image\\aligned\\aligned", f)
        print(file_name)
        file_paths.append(file_name)



    label = label[1]
    image = cv2.imread(file_paths[1])
    image = image[:, :, ::-1]



