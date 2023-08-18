import numpy as np
import torch
from torch import nn

image = torch.rand(1,3,224,224)

proj = nn.Conv2d(3,96,4,4)

print(image.shape)
print(proj(image).shape)
print(proj(image).flatten(2).shape)
print(proj(image).flatten(2).transpose(1,2).shape)