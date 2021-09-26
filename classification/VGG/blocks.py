"""
Normed_VGG Model Blocks 


This model is an upgradation of VGG Model.
As normalization has been added to
this so, named it "Normed_VGG"

© Sagnik Roy, 2021

"""

import torch
import torch.nn as nn
from vgg_config import *


TEST_CONFIG = CONFIG

class Conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, eps=1e-5):
        super(Conv, self).__init__()

        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels, eps),
        )

    def forward(self, x):
        return self.convblock(x)


class Normed_VGG(nn.Module):

    def __init__(self, num_classes, config, h=224, w=224):
        super(Normed_VGG, self).__init__()
        self.blocks = []
        for num_block in range(5):
            block_lev = []
            for layer in config[f"b{num_block + 1}"]:
                block_lev.append(Conv(layer[0][0], layer[0][1], layer[1], 1, layer[1] // 2))
            self.blocks.append(nn.Sequential(*block_lev))

        fc_in = config['b5'][-1][0][1]

        self.model = nn.Sequential(
            self.blocks[0],
            nn.MaxPool2d(2),
            self.blocks[1],
            nn.MaxPool2d(2),
            self.blocks[2],
            nn.MaxPool2d(2),
            self.blocks[3],
            nn.MaxPool2d(2),
            self.blocks[4],
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(fc_in * (h // 32) * (w // 32), 4096),
            nn.Linear(4096, 4096),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
      return self.model(x)


def test():
    batch_size, in_channels, h, w, num_classes = 1, 3, 224, 224, 5
    rand_data = torch.rand(batch_size, in_channels, h, w)
    model = Normed_VGG(num_classes, TEST_CONFIG['A'], h, w)
    assert model(rand_data).shape == (batch_size, num_classes) , 'Model Failed...'
    print('Model Loaded Successfully..')


if __name__ == '__main__':
    test()
