import torch
from torch import nn
import torchvision.models as models
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.model = models.resnet18()
        self.model.fc = nn.Linear(512, 2)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=3, bias=False)

    def forward(self, x):
        x = self.model(x)
        return x

PATH = './car.pth'
net = TheModelClass()
net.load_state_dict(torch.load(PATH))
net.eval()

path = 'D:/data/car_data/ncar.jpg'

img_RGB  = Image.open(path).convert('L')
img_RGB = img_RGB.resize((224, 224))
# torchvision.transforms.ToTensor
tf_toTensor = ToTensor()
# PIL to Tensor
image = tf_toTensor(img_RGB).unsqueeze(dim=0)
print(image.size()) # 3 x 428 x 428


outputs = net(image)
_, predicted = torch.max(outputs, 1)

print(predicted)
