#import

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from imagedata import ImageData
from resnet_simclr import ResNetSimCLR
from utils import info_nce_loss
from FER_220427_SupContrast.data_aug.gaussian_blur import GaussianBlur
from lar_optimizer import LARS

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


#create Network
class Deep_Emotion(nn.Module):
    def __init__(self):
        '''
        Deep_Emotion class contains the network architecture.
        '''
        super(Deep_Emotion,self).__init__()
        self.conv1 = nn.Conv2d(1,10,3)
        self.conv2 = nn.Conv2d(10,10,3)
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv3 = nn.Conv2d(10,10,3)
        self.conv4 = nn.Conv2d(10,10,3)
        self.pool4 = nn.MaxPool2d(2,2)

        self.norm = nn.BatchNorm2d(10)

        self.fc1 = nn.Linear(810,50)
        self.fc2 = nn.Linear(50,7)

        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(640, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 640)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self,input):
        out = self.stn(input)

        out = F.relu(self.conv1(out))
        out = self.conv2(out)
        out = F.relu(self.pool2(out))

        out = F.relu(self.conv3(out))
        out = self.norm(self.conv4(out))
        out = F.relu(self.pool4(out))

        out = F.dropout(out)
        out = out.view(-1, 810)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out

class NN(nn.Module):
    def __init__(self,input_size,num_classes):
        super(NN,self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50,num_classes)

    def forward(self,x):
        x= F.relu(self.fc1(x))
        x= self.fc2(x)

        return x



# model = ResNetSimCLR(base_model="resnet18",out_dim=7)
model =Deep_Emotion()

# model = NN(784,10)
# x = torch.randn(64,784)
# print(model(x).shape)

#Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparmeters
in_channel = 1
num_classes = 7
#learning_rate =0.001
#batch_size = 64
num_epochs = 100
size = 48
#Load Data
train_csvdir= 'C:/Users/1315/Desktop/data/ck_train.csv'
traindir = "C:/Users/1315/Desktop/data/ck_train/"

color_jitter = transforms.ColorJitter(0.8 * 1, 0.8 * 1, 0.8 * 1, 0.2 * 1)
transformation = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomApply([color_jitter], p=0.8),
                                      transforms.RandomGrayscale(p=0.2),
                                      GaussianBlur(kernel_size=int(0.1 * size)),
                                      transforms.ToTensor()])


train_dataset =ImageData(csv_file = train_csvdir, img_dir = traindir, datatype = 'ck_train',transform = transformation)

batch_sizes =[8]
learning_rates = [(0.3 * 8 / 256)]
classes = ['0','1','2','3','4','5','6','7']

for batch_size in batch_sizes:
    for learning_rate in learning_rates:
        step = 0

        # Initialize network
        model = ResNetSimCLR(base_model="resnet18",out_dim=7)
        model.to(device)
        model.train()
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        # loss and optimizer
        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        optimizer= scheduler = LARS(model.parameters(), lr=learning_rate)
        #TODO:공부하기

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
        #                                                        last_epoch=-1)

        writer = SummaryWriter(f'runs/FER_SupContrast/MyNetwork_MiniBatchSize {batch_size} LR {learning_rate}')

        #train
        for epoch in range(num_epochs):
            losses = []
            accuracies =[]

            for batch_idx,(images,targets) in enumerate(train_loader):
                #get data_aug to cuda if possible
               # images = torch.cat(images,dim=0)

                images = images.to(device)
                labels = targets.to(device)

                features = model(images)
                logits,labels = info_nce_loss(batch_size=batch_size,features=features,device=device,real_labels = labels)
                loss = criterion(logits,labels)
                losses.append(loss)

                optimizer.zero_grad()
                loss.backward()

                #gradinet descent or adam step
                optimizer.step()

                #calculate 'running' training accuracy
                img_grid = torchvision.utils.make_grid(images)
               # num_correct = (logits==labels).sum()
               # running_train_acc = float(num_correct)/float(images.shape[0])
               # accuracies.append(running_train_acc)

                #plot things to tensorboard

                writer.add_image('images', img_grid)
#                writer.add_histogram('fc1', model.fc1.weight)
                writer.add_scalar('Training Loss',loss,global_step=step)
              #  writer.add_scalar('Training accuracy',running_train_acc,global_step=step)

                step += 1

            # writer.add_hparams({'lr':learning_rate, 'bsize': batch_size},
            #                    {'accuracy': sum(accuracies) / len(accuracies),
            #                                      'loss': sum(losses) / len(losses)})

            print(f'Mean Loss this epoch was {sum(losses)/len(losses)}')

#check accuarcy on training , test
def check_accuarcy(loader,model):
    if loader.dataset.train:
        print("checking on accuracy on training data_aug")
    else:
        print("checking on accuracy on test data_aug")
    num_correct = 0
    num_sample = 0
    model.eval()

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device = device )
            y = y.to(device=device)

            scores = model(x)
            _,predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_sample += predictions.size(0)

        print(f"got{num_correct}/{num_sample} with accuarcy {float(num_correct)/float(num_sample)*100:.2f}")

    model.train()


# check_accuarcy(train_loader,model)
# check_accuarcy(test_loader,model)
