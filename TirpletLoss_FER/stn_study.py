import torch
import torch.nn.functional as F
from PIL import Image

img = Image.open("C:/Users/1315/Desktop/clean/test/happy_man.jpg")
img.resize((224,224))


x = torch.randn(2, 2,2)
y= torch.randn(2, 2,1)
# feature = torch.randn(2,3,7,7)
print('x',x)
print('y',y)
print(x.size())
print(y.size())
# print('feature',feature)
# thetas = torch.randn(1,6)
# print(thetas)
# theta = thetas[:, (1)*2:(1+1)*2]
# print(theta)
# print(theta.size())
# print(theta.size(0))

theta=torch.cat((x, y), 0)
print('teata',theta)
print(theta.size())
# grid=F.affine_grid(theta, feature.size())
# print(grid)
# print(grid.size())
# xs = F.grid_sample(feature, grid)
# print(xs)
# print(xs.size())


# x = torch.randn(2, 3)
# print(torch.cat((x, x, x), 1).size())
