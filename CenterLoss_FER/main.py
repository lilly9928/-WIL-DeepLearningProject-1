import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from centerloss import CenterLoss
import matplotlib.pyplot as plt
from six.moves import urllib
from dataset import Plain_Dataset
from tqdm import tqdm
from EmotionNetwork import EmotionNetwork

opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.prelu1_2 = nn.PReLU()
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.prelu2_2 = nn.PReLU()
        self.conv3_1 = nn.Conv2d(64, 512, kernel_size=5, padding=2)
        self.prelu3_1 = nn.PReLU()
        self.conv3_2 = nn.Conv2d(512, 512, kernel_size=5, padding=2)
        self.prelu3_2 = nn.PReLU()
        self.preluip1 = nn.PReLU()
        self.ip1 = nn.Linear(512*6*6, 2)
        self.ip2 = nn.Linear(2, 7, bias=False)

    def forward(self, x):
        x = self.prelu1_1(self.conv1_1(x))
        x = self.prelu1_2(self.conv1_2(x))
        x = F.max_pool2d(x,2)
        x = self.prelu2_1(self.conv2_1(x))
        x = self.prelu2_2(self.conv2_2(x))
        x = F.max_pool2d(x,2)
        x = self.prelu3_1(self.conv3_1(x))
        x = self.prelu3_2(self.conv3_2(x))
        x = F.max_pool2d(x,2)
        x = x.view(-1, 512*6*6)
        ip1 = self.preluip1(self.ip1(x))
        ip2 = self.ip2(ip1)

        return ip1, F.log_softmax(ip2, dim=1)

def visualize(feat, labels, epoch):
    plt.ion()
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000']
    plt.clf()
    for i in range(7):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6'], loc = 'upper right')
    plt.xlim(xmin=-8,xmax=8)
    plt.ylim(ymin=-8,ymax=8)
    plt.text(-7.8,7.3,"epoch=%d" % epoch)
    plt.savefig('./images/epoch=%d.jpg' % epoch)
    plt.draw()
    plt.pause(0.001)


def train(epoch):
    print("Training... Epoch = %d" % epoch)
    ip1_loader = []
    idx_loader = []
    for (data, label) in tqdm(train_loader):
        data = data.to(device)
        label = label.to(device)

        ip1,pred = model(data)
        loss = nllloss(pred, label) + loss_weight * centerloss(label, ip1)

        optimizer4nn.zero_grad()
        optimzer4center.zero_grad()

        loss.backward()

        optimizer4nn.step()
        optimzer4center.step()

        ip1_loader.append(ip1)
        idx_loader.append((label))

    feat = torch.cat(ip1_loader, 0)
    labels = torch.cat(idx_loader, 0)
    visualize(feat.data.cpu().numpy(),labels.data.cpu().numpy(),epoch)

if __name__ == '__main__':

    use_cuda = torch.cuda.is_available() and True
    device = torch.device("cuda" if use_cuda else "cpu")
    # Dataset
    fileroot = 'C:/Users/1315/Desktop/data'
    traincsv_file = fileroot + '/' + 'ck_train.csv'
    # valcsv_file = fileroot + '/' + 'test_ck.csv'
    # testcsv_file = fileroot + '/' + 'finaltest.csv'
    train_img_dir = fileroot + '/' + 'ck_train/'
    # val_img_dir = fileroot + '/' + 'test_ck/'
    # test_img_dir = fileroot + '/' + 'finaltest/'

    transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = Plain_Dataset(csv_file=traincsv_file, img_dir=train_img_dir, datatype='ck_train',
                                  transform=transformation)
    # val_dataset = Plain_Dataset(csv_file=valcsv_file, img_dir=val_img_dir, datatype='test_ck', transform=transformation)
    # test_dataset = Plain_Dataset(csv_file=testcsv_file, img_dir=test_img_dir, datatype='finaltest',
    #                              transform=transformation)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=0)
    # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=0)

    # Model
    model = Net().to(device)

    # NLLLoss
    nllloss = nn.NLLLoss().to(device) #CrossEntropyLoss = log_softmax + NLLLoss
    # CenterLoss
    loss_weight = 1
    centerloss = CenterLoss(10, 2).to(device)

    # optimzer4nn
    optimizer4nn = optim.SGD(model.parameters(),lr=0.001,momentum=0.9, weight_decay=0.0005)
    sheduler = lr_scheduler.StepLR(optimizer4nn,20,gamma=0.8)

    # optimzer4center
    optimzer4center = optim.SGD(centerloss.parameters(), lr = 0.5)

    for epoch in range(1):
        sheduler.step()
        # print optimizer4nn.param_groups[0]['lr']
        train(epoch+1)


