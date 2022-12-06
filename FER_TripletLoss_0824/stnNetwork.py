import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Network(nn.Module):
    def __init__(self):
        '''
        Deep_Emotion class contains the network architecture.
        '''
        super(Network, self).__init__()
        self.Resnet50 = models.resnet50()
        self.Resnet50.fc = nn.Linear(512, 7)

        self.localization = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(24, 12, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(8*8*12, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

    def stn(self, x):
        xs = self.localization(x)
        xs = F.dropout(xs)
        xs = xs.view(-1, 8*8*12)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        x = self.stn(x)
        out = self.Resnet50(x)

        return out


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)