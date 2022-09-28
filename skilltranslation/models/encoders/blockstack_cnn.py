from skilltranslation.models.encoders.base import Encoder
from skilltranslation.models.encoders.cnn import CNNEncoder
import torch.nn.functional as F
import torch.nn as nn
import torch
class BlockStackCNNEncoder(CNNEncoder):
    def __init__(self, channels=[32, 32], input_shape=(3, 128, 128), kernel_size=5, hidden_sizes=[128, 128]) -> None:
        super().__init__(channels=channels, input_shape=input_shape, kernel_size=kernel_size, hidden_sizes=hidden_sizes)
        self.out_dims = 25 + hidden_sizes[-1]

    def forward(self, x):
        batch_size = x.shape[0]
        # x - (B, 25 + input_shape_size)
        imgs = x[:, 25:] # (B, input_shape_size)
        imgs = imgs.reshape(batch_size, *self.input_shape)
        states = x[:, :25] # (B, 25)
        # x - (B, C, H, W)
        
        imgs = self.cnn(imgs)
        imgs = imgs.view(batch_size, -1)
        imgs = self.mlp(imgs)
        # import pdb;pdb.set_trace()
        # x - (B, H)
        x = torch.hstack([imgs, states]) # (B, H+25)
        return x
