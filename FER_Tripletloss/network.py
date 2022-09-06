import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchsummary import summary

class Network(nn.Module):
    def __init__(self):

        super(Network, self).__init__()
        #self.Resnet18 = resnet18()
        self.Resnet18 = models.resnet18(pretrained=True)
        self.Resnet18.fc =nn.Linear(512,7)
        #self.Resnet18.fc = nn.Identity()
        self.Resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=3, bias=False)

        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(110, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        # self.fc_loc[2].weight.data.zero_()
        #self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        xs = F.dropout(xs)
        print(xs.shape)
        xs = xs.view(-1,110)
        print(xs.shape)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(),align_corners=True)
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x1,x2,x3):
       # x = self.stn(x)
       out1 = self.Resnet18(x1)
       out2 = self.Resnet18(x2)
       out3 = self.Resnet18(x3)

       return out1,out2,out3


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Network().to(device)
summary(model, input_size=[(1, 448, 448),(1,448,448),(1,448,448)])
