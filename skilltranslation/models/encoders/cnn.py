from skilltranslation.models.encoders.base import Encoder
import torch.nn.functional as F
import torch.nn as nn
class CNNEncoder(Encoder):
    def __init__(self, channels=[32, 32], input_shape=(3, 128, 128), kernel_size=5, hidden_sizes=[128, 128]) -> None:
        super().__init__(out_dims=hidden_sizes[-1], input_shape=input_shape)
        in_channels = channels[0]
        layers = []
        self.input_shape = input_shape
        self.channels = [self.input_shape[0]] + channels
        H, W = self.input_shape[1], self.input_shape[2]
        for i in range(len(self.channels) - 1):
            in_c = self.channels[i]
            out_c = self.channels[i + 1]
            conv_layer = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=kernel_size, padding="same")
            layers += [conv_layer, nn.ReLU(), nn.MaxPool2d(2, 2)]
        self.cnn = nn.Sequential(*layers)
        layers = []
        self.hidden_sizes = [H * W * self.channels[-1] // (2 ** ((len(self.channels) - 1)*2))] + hidden_sizes
        for i in range(len(self.hidden_sizes) - 1):
            layers += [nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i+1])]
            # if i <= len(self.hidden_sizes) - 2:
            layers += [nn.ReLU()]
        self.mlp = nn.Sequential(*layers)
    def forward(self, x):
        # x - (B, C, H, W)
        batch_size = x.shape[0]
        x = self.cnn(x)
        x = x.view(batch_size, -1)
        x = self.mlp(x)
        # x - (B, H)
        return x
