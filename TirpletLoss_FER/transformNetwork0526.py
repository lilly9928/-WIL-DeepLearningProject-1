import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import resnet18
from torchsummary import summary

class Network(nn.Module):
    def __init__(self):
        '''
        Deep_Emotion class contains the network architecture.
        '''
        super(Network, self).__init__()
        self.Resnet18 = resnet18()

        self.localization = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(24, 30, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(8*8*30, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

    def stn(self, x):
        xs = self.localization(x)
        xs = F.dropout(xs)
        xs = xs.view(-1, 8*8*30)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        out = self.stn(x)
        #out = self.Resnet18(x)

        return out


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Network().to(device)
summary(model, input_size=(1, 224, 224))
