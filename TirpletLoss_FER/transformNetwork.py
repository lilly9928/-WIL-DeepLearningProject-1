import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import resnet18
import torchvision.models as models
from torchsummary import summary

class Network(nn.Module):
    def __init__(self):

        super(Network, self).__init__()
        self.Resnet18 = models.resnet18(pretrained=True)
        #self.Resnet18.fc = nn.Identity()
        self.Resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=3, bias=False)
        #self.Resnet18 = models.resnet18(pretrained=True)
        self.Resnet18.fc =nn.Linear(512,7)
        #self.Resnet18.fc = nn.Identity()
        #self.Resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=3, bias=False)

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(512, 50)
        self.fc2 = nn.Linear(50, 7)

        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
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
        xs = F.dropout(xs)
        xs = xs.view(-1,640)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        x = self.stn(x)
        out = self.Resnet18(x)


        return out



def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Network().to(device)
summary(model, input_size=(1, 48, 48))

resnet18()
