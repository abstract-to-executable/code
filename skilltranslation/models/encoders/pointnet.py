import torch
from skilltranslation.models.encoders.base import Encoder
import torch.nn.functional as F
import torch.nn as nn
class PointNetEncoder(Encoder):
    def __init__(self, channels=[32, 32], global_channels=[32,32], input_shape=(60, 3), hidden_sizes=[128, 128], input_offset=33) -> None:
        super().__init__(out_dims=hidden_sizes[-1], input_shape=input_shape)
        self.input_offset = input_offset
        in_channels = channels[0]
        layers = []
        self.input_shape = input_shape
        self.channels = [self.input_shape[1]] + channels
        for i in range(len(self.channels) - 1):
            layers += [nn.Conv1d(self.channels[i], self.channels[i+1], 1)]
            layers += [nn.ReLU()]
        self.mlp_1 = nn.Sequential(*layers)

        layers = []
        self.global_channels = [self.channels[-1]] + global_channels
        for i in range(len(self.global_channels) - 1):
            layers += [nn.Conv1d(self.global_channels[i], self.global_channels[i+1], 1)]
            layers += [nn.ReLU()]
        self.mlp_2 = nn.Sequential(*layers)

        layers = []
        self.hidden_sizes = [self.global_channels[-1] + self.input_offset] + hidden_sizes
        for i in range(len(self.hidden_sizes) - 1):
            layers += [nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i+1])]
            # if i <= len(self.hidden_sizes) - 2:
            layers += [nn.ReLU()]
        self.mlp_3 = nn.Sequential(*layers)
    def forward(self, x):
        # x - (B, offset + n*3)
        batch_size = x.shape[0]
        rest = x[:, self.input_offset:] # (B, input_shape_size)
        rest = rest.reshape(batch_size, *self.input_shape).permute(0,2,1) # (B, n, 3)
        states = x[:, :self.input_offset] # (B, offset)
        
        # print("rest",rest.shape)
        pt_feats = self.mlp_1(rest)
        # print("ptf",pt_feats.shape)
        
        global_feats = self.mlp_2(pt_feats)
        global_feats = torch.max(global_feats, dim=2)[0]
        # print("gfeats", global_feats.shape)
        x = torch.cat([
            states,
            # pt_feats,
            global_feats
        ], dim =1) # (B, s+g)
        x = x.view(batch_size, -1)
        x = self.mlp_3(x)
        # x - (B, H)
        return x