import numpy as np
import torch.nn as nn
import torchvision.models as models

class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}

        self.backbone = self._get_basemodel(base_model)
        self.lbp_backbone = self.backbone
        self.lbp_backbone.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=3, bias=False)

        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

        # add mlp projection head
        # self.lbp_backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
          print('')
        else:
            return model

    def forward(self, x,lbp):
        image_out=self.backbone(x)
        lbp_out = self.lbp_backbone(lbp)

        return lbp_out*image_out