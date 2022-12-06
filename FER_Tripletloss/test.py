from torchvision import transforms
from PIL import Image
from PIL import ImageFile
import torch
import torch.nn.functional as F

ImageFile.LOAD_TRUNCATED_IMAGES = True

im_path = 'C:/Users/1315/Desktop/clean/data/ck_train/ck_train0.jpg'
image = Image.open(im_path)
image = image.convert('RGB')

im_transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor()
])
image = im_transform(image)
# Add batch dimension
image = image.unsqueeze(dim=0)

d = torch.linspace(-1, 1, 224)
meshx, meshy = torch.meshgrid((d, d))

# Just to see the effect
meshx = meshx * 0.3
meshy = meshy * 0.9

grid = torch.stack((meshy, meshx), 2)
grid = grid.unsqueeze(0)
warped = F.grid_sample(image, grid, mode='bilinear', align_corners=False)

to_image = transforms.ToPILImage()
to_image(image.squeeze()).show()
to_image(warped.squeeze()).show(title='WARPED')