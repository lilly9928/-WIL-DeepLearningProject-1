# import numpy as np
# import torch
import torch.nn as nn
import torch
# import torch.nn.functional as F
# import math, copy, time
# from torch.autograd import Variable
# import matplotlib.pyplot as plt
# import seaborn
# seaborn.set_context(context="talk")
#
# class EncoderDecoder(nn.Module):
#     def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
#         super(EncoderDecoder, self).__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#         self.src_embed = src_embed
#         self.tgt_embed = tgt_embed
#         self.generator = generator
#
#     def forward(self,src, tgt, src_mask, tgt_mask):
#         return self.decode(self.encode(src, src_mask), src_mask,
#                            tgt, tgt_mask)
#
#     def encode(self, src, src_mask):
#         return self.encoder(self.src_embed(src), src_mask)
#
#     def decode(self, memory, src_mask, tgt, tgt_mask):
#         return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
#
# class Generator(nn.Module):
#     "Define standard linear + softmax generation step."
#     def __init__(self, d_model, vocab):
#         super(Generator, self).__init__()
#         self.proj = nn.Linear(d_model, vocab)
#
#     def forward(self, x):
#         return F.log_softmax(self.proj(x), dim=-1)


transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
src = torch.rand((10, 32, 512)) # the sequence to the encoder
tgt = torch.rand((20, 32, 512)) # sequence to the decoder
out = transformer_model(src, tgt)

print(src.shape)
print(tgt.shape)
print(out.shape)

decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
