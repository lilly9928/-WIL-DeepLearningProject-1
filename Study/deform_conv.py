import torchvision.ops
import torch

input = torch.rand(4,3,244,244)
kh,kw = 10,10
max_offset = max(kh,kw)/4.
#나누기 4를 하는 의미??

weight = torch.rand(5,3,kh,kw)
offset = torch.rand(4,2*kh*kw,235,235).clamp(-max_offset,max_offset)
mask = torch.rand(4,kh*kw,235,235)

out = torchvision.ops.deform_conv2d(input, offset, weight, mask=mask)

print(out.shape)