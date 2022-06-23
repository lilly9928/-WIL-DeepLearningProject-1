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
from ViTStudy_PatchEmbedding import PatchEmbedding

#라이브러리와 프레임워크

input = torch.randn(8,3,224,224)

x = PatchEmbedding()(input)
print('---------------------------')
#Linear Projection
emb_size = 768
num_heads = 8

keys = nn.Linear(emb_size,emb_size)
queries = nn.Linear(emb_size,emb_size)
values = nn.Linear(emb_size,emb_size)
print(keys)
print(queries)
print(values)


queries = rearrange(queries(x), "b n (h d) -> b h n d", h=num_heads)
keys = rearrange(keys(x), "b n (h d) -> b h n d", h=num_heads)
values  = rearrange(values(x), "b n (h d) -> b h n d", h=num_heads)

print('shape(BATCH, HEADS, SEQUENCE_LEN, EMBEDDING_SIZE.) :', queries.shape, keys.shape, values.shape)
print('---------------------------')
#Scaled Dot Product Attention
# Queries * Keys
energy = torch.einsum('bhqd, bhkd -> bhqk',queries,keys)
print('energy (BATCH, HEADS, QUERY_LEN, KEY_LEN) :',energy.shape)

#Get Attention Score
scaling = emb_size ** (1/2)
att = F.softmax(energy, dim=-1)/scaling
print('att:' , att.shape)

# Attention Score * values
out = torch.einsum('bhal, bhlv -> bhav ', att, values)
print('out :', out.shape)

# Rearrage to emb_size
out = rearrange(out, "b h n d -> b n (h d)")
print('out2 : ', out.shape)


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


