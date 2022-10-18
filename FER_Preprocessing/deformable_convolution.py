import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import numpy as np
import random
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torchvision.ops
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from imagedata import ImageData
import torchvision.models as models


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)


class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):
        super(DeformableConv2d, self).__init__()

        self.padding = padding

        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size * kernel_size,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels,
                                        1 * kernel_size * kernel_size,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=self.padding,
                                        bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x):
        h, w = x.shape[2:]
        max_offset = max(h, w) / 4.

        offset = self.offset_conv(x).clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))

        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias,
                                          padding=self.padding,
                                          mask=modulator
                                          )
        return x

class Modify_Resnet(nn.Module):
    def __init__(self):
        super(Modify_Resnet,self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.layer3.conv1 = DeformableConv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        #self.model.layer3.conv2 = DeformableConv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)
       # self.model.layer1.conv1 = DeformableConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
       # self.model.layer2.conv1 = DeformableConv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
       # self.model.layer2.conv2 = DeformableConv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.model.fc = nn.Linear(512, 7)

        self.deconv1 = DeformableConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.deconv2 = DeformableConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.deconv3 = DeformableConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)

        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 7)



    def forward(self,x):
      #  x = torch.relu(self.deconv1(x))
      #  x = torch.relu(self.deconv2(x))

        x = self.model(x)
        # x = self.gap(x)
        # x = x.flatten(start_dim=1)
        # x = self.fc(x)

        return x


class MNISTClassifier(nn.Module):
    def __init__(self,
                 deformable=False):
        super(MNISTClassifier, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        conv = nn.Conv2d if deformable == False else DeformableConv2d
        self.conv4 = conv(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5 = conv(32, 32, kernel_size=3, stride=1, padding=1, bias=True)

        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 7)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)  # [14, 14]
        x = torch.relu(self.conv2(x))
        x = self.pool(x)  # [7, 7]
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = self.gap(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x



def train(model, loss_function, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()



def val(model, loss_function, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    num_data = 0
    with torch.no_grad():
        for data, target in test_loader:
            org_data, target = data.to(device), target.to(device)

            for scale in np.arange(0.5, 1.6, 0.1):  # [0.5, 0.6, ... ,1.2, 1.3, 1.4, 1.5]
                data = transforms.functional.affine(org_data, scale=scale, angle=0, translate=[0, 0], shear=0)
                output = model(data)
                test_loss += loss_function(output, target).item()  # sum up batch mean loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                num_data += len(data)

    test_loss /= num_data

    test_acc = 100. * correct / num_data
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct, num_data,test_acc))
    return test_acc



def main():
    # Training settings
    seed = 1
    setup_seed(seed)

    use_cuda = torch.cuda.is_available()
    batch_size = 12
    lr = 1e-3
    gamma = 0.7
    epochs = 100

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 4,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_transform = transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load Data
    train_csvdir = 'C:/Users/1315/Desktop/data/ck_train.csv'
    traindir = "C:/Users/1315/Desktop/data/ck_train/"
    val_csvdir = 'C:/Users/1315/Desktop/data/ck_val.csv'
    valdir = "C:/Users/1315/Desktop/data/ck_val/"

    train_dataset = ImageData(csv_file=train_csvdir, img_dir=traindir, datatype='ck_train', transform=transform)
    val_dataset = ImageData(csv_file=val_csvdir, img_dir=valdir, datatype='ck_val', transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)


    model = Modify_Resnet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    loss_function = nn.CrossEntropyLoss()
    best_test_acc = 0.
    for epoch in range(1, epochs + 1):
        print(epoch)
        train(model, loss_function, device, train_loader, optimizer, epoch)
        best_test_acc = max(best_test_acc, val(model, loss_function, device, val_loader))
        scheduler.step()
    print("best top1 acc(%): ", f"{best_test_acc:.2f}")
    torch.save(model.state_dict(), './models/dcn_weights_resnet.pt')


if __name__ == '__main__':
    #print(models.resnet18())
    main()