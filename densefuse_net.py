# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 21:03:03 2019

@author: win10
"""

import torch.nn as nn
import torch
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential()
        self.layers.add_module('Conv2', nn.Conv2d(64, 64, 3, 1, 1))
        self.layers.add_module('Act2', nn.ReLU(inplace=True))
        self.layers.add_module('Conv3', nn.Conv2d(64, 32, 3, 1, 1))
        self.layers.add_module('Act3', nn.ReLU(inplace=True))
        self.layers.add_module('Conv4', nn.Conv2d(32, 16, 3, 1, 1))
        self.layers.add_module('Act4', nn.ReLU(inplace=True))
        self.layers.add_module('Conv5', nn.Conv2d(16, 1, 3, 1, 1))

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.Conv1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.Relu = nn.ReLU(inplace=True)

        self.layers = nn.ModuleDict({
            'DenseConv1': nn.Conv2d(16, 16, 3, 1, 1),
            'DenseConv2': nn.Conv2d(32, 16, 3, 1, 1),
            'DenseConv3': nn.Conv2d(48, 16, 3, 1, 1)
        })

    def forward(self, x):
        x = self.Relu(self.Conv1(x))
        for i in range(len(self.layers)):
            out = self.layers['DenseConv' + str(i + 1)](x)
            x = torch.cat([x, out], 1)
        return x


class DenseFusionNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x1, x2, fuse_strategy='average', window_width=1):
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)
        if fuse_strategy == 'average':
            x = (x1 + x2) / 2
        elif fuse_strategy == 'l1':
            activity_map1 = x1.abs()
            activity_map2 = x2.abs()

            kernel = torch.ones(2 * self.window_width + 1, 2 * self.window_width + 1) / (2 * self.window_width + 1) ** 2
            kernel = kernel.to(x1.device).type(torch.float32)[None, None, :, :]
            kernel = kernel.expand(x1.shape[1], x1.shape[1], 2 * self.window_width + 1, 2 * self.window_width + 1)
            activity_map1 = F.conv2d(activity_map1, kernel, padding=self.window_width)
            activity_map2 = F.conv2d(activity_map2, kernel, padding=self.window_width)
            weight_map1 = activity_map1 / (activity_map1 + activity_map2)
            weight_map2 = activity_map2 / (activity_map1 + activity_map2)
            x = weight_map1 * x1 + weight_map2 * x2
        else:
            raise RuntimeError(f'fuse_strategy: {fuse_strategy} is not supported!')
        return self.decoder(x)
