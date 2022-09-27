import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
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
import matplotlib.pyplot as plt
import torchvision


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
        src =cv2.imread(self.img_dir + self.datatype + str(idx) + '.jpg')
        img = cv2.resize(src, (244, 244))
        lables = np.array(self.lables[idx])
        lables = torch.from_numpy(lables).long()

        if self.transform:
            img = self.transform(img)
            #img = self.transform(hog_image_rescaled)

        return img,lables


if __name__ == "__main__":
   #check sample images
    def show(img, y=None):
        npimg = img.numpy()
        npimg_tr = np.transpose(npimg, (1, 2, 0))
        plt.imshow(npimg_tr)

        if y is not None:
            plt.title('labels:' + str(y))

#compound
    # classes = ['Happily Surprised', 'Happily Disgusted', 'Sadly Fearful', 'Sadly Angry', 'Sadly Surprised',
    #            'Sadly Disgusted', 'Fearfully Angry',
    #            'Fearfully Surprised', 'Angrily Surprised', 'Angrily Disgusted', 'Disgustedly Surprised']

   #basic
    classes = [ 'Surprise', 'Fear','Disgust','Happiness','Sadness','Anger','Neutral']

    train_csvdir = 'D:/data/FER/ck_images/ck_train.csv'
    traindir = "D:/data/FER/ck_images/FourierImages/ck_train"

    train_transforms = transforms.Compose([
       # transforms.ToPILImage(),
       # # transforms.Resize((224, 224)),
       #  transforms.Resize((224, 224)),
        transforms.ToTensor()
       # transforms.Normalize(mean=[0.485, 0.456, 0.406],
       #                      std=[0.229, 0.224, 0.225]),
       # transforms.RandomErasing(scale=(0.02, 0.25))
    ])

    dataset=ImageData(csv_file = train_csvdir, img_dir = traindir, datatype = 'ck_train',transform = train_transforms)

    grid_size=4
    rnd_ind = np.random.randint(0, len(dataset), grid_size)

    print(dataset[rnd_ind[1]])

    x_grid = [dataset[i][0] for i in rnd_ind]
    y_grid = [classes[dataset[i][1]] for i in rnd_ind]

    x_grid = torchvision.utils.make_grid(x_grid, nrow=grid_size, padding=2)
    plt.figure(figsize=(10,10))
    show(x_grid, y_grid)
    plt.show()