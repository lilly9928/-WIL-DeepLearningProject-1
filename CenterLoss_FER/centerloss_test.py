import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from centerloss import CenterLoss
from dataset import Plain_Dataset
import torchvision
from tqdm import tqdm

from EmotionNetwork import EmotionNetwork

from torch.utils.tensorboard import SummaryWriter

# opener = urllib.request.build_opener()
# opener.addheaders = [('User-agent', 'Mozilla/5.0')]
# urllib.request.install_opener(opener)

if __name__ == '__main__':

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


    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Model
   # model = Net().to(device)
    model = EmotionNetwork().to(device)

    #hyperparmeters
    in_channel = 1
    num_classes = 7
    num_epochs = 5
    learning_rate = 0.001
    batch_size = 64


    # Dataset
    fileroot = 'C:/Users/1315/Desktop/data_aug'
    traincsv_file = fileroot + '/' + 'ck_train.csv'
    train_img_dir = fileroot + '/' + 'ck_train/'


    transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = Plain_Dataset(csv_file=traincsv_file, img_dir=train_img_dir, datatype='ck_train',
                                  transform=transformation)


    batch_sizes = [8,16,256]
    learning_rates = [0.01,0.001]

    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            step = 0

            model = Net()
            model.to(device)
            model.train()
            # Load Data
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
            # NLLLoss
            nllloss = nn.NLLLoss().to(device)  # CrossEntropyLoss = log_softmax + NLLLoss
            # CenterLoss
            loss_weight = 1
            centerloss = CenterLoss(10, 2).to(device)
            # optimzer4nn
            optimizer4nn = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
            sheduler = lr_scheduler.StepLR(optimizer4nn, 20, gamma=0.8)
            # optimzer4center
            optimzer4center = optim.SGD(centerloss.parameters(), lr=0.5)
            writer = SummaryWriter(f'runs/CenterLoss_FER/MiniBatchSize {batch_size} LR {learning_rate}')

            for epoch in range(num_epochs):
                losses = []
                accuracies = []
                ip1_loader = []
                idx_loader = []

                for (data, label) in tqdm(train_loader):
                    # get data_aug to cuda if possible
                    data = data.to(device=device)
                    label = label.to(device=device)

                    _, pred = model(data)
                    loss = nllloss(pred, label) + loss_weight * centerloss(label, model.fc2)
                    losses.append(loss.item())

                    optimizer4nn.zero_grad()
                    optimzer4center.zero_grad()

                    loss.backward()

                    optimizer4nn.step()
                    optimzer4center.step()

                    ip1_loader.append(model.fc2)
                    idx_loader.append(label)

                 # calculate 'running' training accuracy
                features = data.reshape(data.shape[0], -1)
                writer.add_scalar('Training Loss', loss, global_step=step)

                step += 1

                print(f'Mean Loss this epoch was {sum(losses) / len(losses)}')



