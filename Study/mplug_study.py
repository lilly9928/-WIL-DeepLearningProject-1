import argparse
import os
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

# torch.manual_seed(5)
# print(torch.rand(5))
# torch.manual_seed(5)
# print(torch.rand(5))
# torch.manual_seed(5)
# print(torch.rand(5))
# print(torch.rand(5))


# it = iter(range(3))
# print(it)
#
# print(next(it))
# print(next(it))
# print(next(it))


from sentence_transformers import SentenceTransformer, util
from PIL import Image

#Load CLIP model
model = SentenceTransformer('clip-ViT-L-14')

#Encode an image:
img_emb = model.encode(Image.open('two_dogs_in_snow.jpg'))

#Encode text descriptions
text_emb = model.encode(['Two dogs in the snow', 'A cat on a table', 'A picture of London at night'])

#Compute cosine similarities
cos_scores = util.cos_sim(img_emb, text_emb)
print(cos_scores)