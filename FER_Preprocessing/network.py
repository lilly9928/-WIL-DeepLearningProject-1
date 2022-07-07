import torch.nn as nn
import torchvision.models as models

class Resnet(nn.Module):

    def __init__(self,base_model,out_dim):
        super(Resnet, self).__init__()
        self.resnet_dict = {"resnet18":models.resnet18(pretrained=False,num_classes = out_dim),
                            "resnet50": models.resnet50(pretrained=False,num_classes= out_dim)}
        self.backbone = self._get_basemodel(base_model)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=3, bias=False)


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


