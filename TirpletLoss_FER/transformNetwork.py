import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import resnet18
import torchvision.models as models
from torchsummary import summary

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicStn(nn.Module):

    def __init__(self, parallel, in_feature, **kwargs):
        super(BasicStn, self).__init__()
        self.conv = conv1x1(in_feature, 128)
        self.fc_loc = nn.Sequential(
            nn.Linear(128*7*7, 64),
            nn.Tanh(),
            nn.Linear(64, 2*len(parallel)),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv(x)
       # print(x.shape)
        x = x.view(-1, 128*7*7)
        x = self.fc_loc(x)
        return x


class BasicFc(nn.Module):

    def __init__(self, in_feature, out_feature, p=0, **kwargs):
        super(BasicFc, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p)
        self.fc = nn.Linear(in_feature, out_feature)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class StnFc975(nn.Module):

    def __init__(self, parallel, in_feature, out_feature):
        super(StnFc975, self).__init__()
        self.parallel = parallel
        self.out_feature = out_feature
        self.stn = BasicStn(parallel, in_feature)
        self.fc1 = BasicFc(in_feature, out_feature)
        self.fc2 = BasicFc(in_feature, out_feature)
        self.fc3 = BasicFc(in_feature, out_feature)
        self.fc4 = BasicFc(in_feature, out_feature)

    def forward(self, feature):
        x = self.fc1(feature)
        thetas = self.stn(feature)
        i = 0
        theta = thetas[:, (i)*2:(i+1)*2] #thetas => feature 추출 값 n..?
        theta = theta.view(-1, 2, 1)
        crop_matrix = torch.tensor([[self.parallel[i], 0], [0, self.parallel[i]]], dtype=torch.float).cuda()
        crop_matrix = crop_matrix.repeat(theta.size(0), 1).reshape(theta.size(0), 2, 2)
        theta = torch.cat((crop_matrix, theta), dim=2)
        grid = F.affine_grid(theta, feature.size())
        xs = F.grid_sample(feature, grid)
        x += self.fc2(xs)
        i += 1

        theta = thetas[:, (i)*2:(i+1)*2]
        theta = theta.view(-1, 2, 1)
        crop_matrix = torch.tensor([[self.parallel[i], 0], [0, self.parallel[i]]], dtype=torch.float).cuda()
        crop_matrix = crop_matrix.repeat(theta.size(0), 1).reshape(theta.size(0), 2, 2)
        theta = torch.cat((crop_matrix, theta), dim=2)
        grid = F.affine_grid(theta, feature.size())
        xs = F.grid_sample(feature, grid)
        x += self.fc3(xs)
        i += 1

        theta = thetas[:, (i)*2:(i+1)*2]
        theta = theta.view(-1, 2, 1)
        crop_matrix = torch.tensor([[self.parallel[i], 0], [0, self.parallel[i]]], dtype=torch.float).cuda()
        crop_matrix = crop_matrix.repeat(theta.size(0), 1).reshape(theta.size(0), 2, 2)
        theta = torch.cat((crop_matrix, theta), dim=2)
        grid = F.affine_grid(theta, feature.size())
        xs = F.grid_sample(feature, grid)
        x += self.fc4(xs)

        return x


class ResNetMultiStn(nn.Module):

    def __init__(self, block, layers, num_classes=7, zero_init_residual=False, p=0, parallel=[0.9, 0.7, 0.5]):
        super(ResNetMultiStn, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # [64, 112, 112]
        self.layer1 = self._make_layer(block, 64, layers[0])
        # [64, 112, 112]
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # [128, 56, 56]
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # [256, 28, 28]
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # [512, 14, 14]
        self.stn_fc = StnFc975(parallel, 512 * block.expansion, num_classes)
        #         self.stn_fc = StnFc8642(parallel, 512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) #64,112,112

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feature = self.layer4(x) #512

        x = self.stn_fc(feature)

        return feature,x

# class Network(nn.Module):
#     def __init__(self):
#
#         super(Network, self).__init__()
#         self.Resnet18 = models.resnet34(pretrained=True)
#         #self.Resnet18.fc = nn.Identity()
#         self.Resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=3, bias=False)
#         #self.Resnet18.fc =nn.Linear(512,7)
#         self.Resnet18.fc = nn.Identity()
#         #self.Resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=3, bias=False)
#
#         self.fc1 = nn.Linear(512, 50)
#         self.fc2 = nn.Linear(50, 7)
#
#         self.localization = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=50, kernel_size=7),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True),
#             nn.Conv2d(50, 100, kernel_size=5),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True),
#         )
#
#         self.fc_loc = nn.Sequential(
#             nn.Linear(100*52*52, 100),
#             nn.ReLU(True),
#             nn.Linear(100, 3 * 2)
#         )
#         self.fc_loc[2].weight.data.zero_()
#         self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
#
#     def stn(self, x):
#         xs = self.localization(x)
#         xs = F.dropout(xs)
#        # print(xs.shape)
#         xs = xs.view(-1,100*52*52)
#        # print(xs.shape)
#         theta = self.fc_loc(xs)
#         theta = theta.view(-1, 2, 3)
#
#         grid = F.affine_grid(theta, x.size())
#         x = F.grid_sample(x, grid)
#
#         return x
#
#     def forward(self, x):
#         x = self.stn(x)
#         feature = self.Resnet18(x)
#         out = self.fc1(feature)
#         out = self.fc2(out)
#
#         return feature,out



def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNetMultiStn(Bottleneck, [3, 4, 6, 3]).to(device)
#summary(model, input_size=(3, 224, 224))

