import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Resnet(nn.Module):

    def __init__(self,base_model,out_dim):
        super(Resnet, self).__init__()
        self.resnet_dict = {"resnet18":models.resnet18(pretrained=False,num_classes = out_dim),
                            "resnet34": models.resnet34(pretrained=False,num_classes= out_dim)}
        self.backbone = self._get_basemodel(base_model)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=3, bias=False)


    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            print('')
        else:
            return model


    def forward(self, x):
        image_out = self.backbone(x)

        return image_out

class CropNetwork(nn.Module):

    def __init__(self,input_size=(224,224)):
        super(CropNetwork,self).__init__()

        self.fc1 = nn.Linear(input_size[0],input_size[0]/2)
        self.fc2 = nn.Linear(input_size[0]/2, (input_size[0]/2)/2)
        self.fc3 = nn.Linear((input_size[0]/2)/2, 4)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.relu(self.fc3(x))

        return x



