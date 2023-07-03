
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose,Resize,ToTensor
from einops import rearrange,reduce,repeat
from einops.layers.torch import Rearrange,Reduce
from torchsummary import summary

#라이브러리와 프레임워크

x = torch.randn(8,1,224,224)

# 배치 사이즈 8 ,채널 3 , h,w = 244,244
# batch x channel x height x width -> batch x channel x patch number x ( patch size *patch size * channel)
# patch number = (height x width / patch size x patch size )

patch_size = 16
# 16픽셀

print('x: ', x.shape)
patches = rearrange(x, 'b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size)
print('patches:', patches.shape)
# 테스트

patch_size = 16
in_channels = 1
emb_size = 256 # patch_size

projection = nn.Sequential(nn.Conv2d(in_channels,emb_size,kernel_size=patch_size,stride=patch_size),Rearrange('b e (h) (w) -> b (h w) e '))
print('usedCov2d_patches:',projection(x).shape)

# cls Token , Positional Encoding 추가
emb_size = 768
img_size = 224
patch_size = 16
print('----------------------')
#이미지를 패치 사이즈로 나누고 flatten
projected_x = projection(x)
print('Projected X shape : ', projected_x.shape)

#cls_token , pos encoding Parameter 정의
cls_token = nn.Parameter(torch.randn(1,1,emb_size)) #number placed in from of each sequence
position = nn.Parameter(torch.randn((img_size//patch_size) **2 + 1, emb_size)) #spatial information.
print('Cls Shape: ', cls_token.shape, 'Pos Shape:', position.shape)

#cls_token 반복하여 배치 사이즈의 크기 맞추기
batch_size = 8
cls_tokens = repeat(cls_token,'() n e -> b n e',b = batch_size)
print('Repeat Cls shape: ', cls_tokens.shape)

#cls_token  과 projected_x 를 concatenate
cat_x = torch.cat([cls_tokens,projected_x],dim = 1)

#postion encoding 을 더해줌
cat_x += position
print('output: ', cat_x.shape)

#PatchEmbedding class 구현
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += self.positions

        return x

