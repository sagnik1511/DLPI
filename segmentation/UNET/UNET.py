import torch
import torch.nn as nn
import warnings
import torchvision.transforms as transforms
from torchsummary import summary

warnings.filterwarnings('ignore')

def copy_and_crop(size):
    return transforms.CenterCrop(size = size)


class Conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size = (3, 3), stride = 1, padding = 0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        return self.conv(x)

class up_block(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor = 2, kernel_size = (1, 1), stride = 1):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor = scale_factor),
            nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride),
        )
    def forward(self, x):
        return self.upsample(x)

class conv_block(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cnnblock = nn.Sequential(
            Conv(in_channels, out_channels),
            nn.ReLU(),
            Conv(out_channels, out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.cnnblock(x)


class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.c1 = conv_block(in_channels, 64)
        self.c2 = conv_block(64, 128)
        self.c3 = conv_block(128, 256)
        self.c4 = conv_block(256, 512)
        self.c5 = conv_block(512, 1024)
        self.c6 = conv_block(1024, 512)
        self.c7 = conv_block(512, 256)
        self.c8 = conv_block(256, 128)
        self.c9 = conv_block(128, 64)
        self.out = nn.Conv2d(64, num_classes, (1, 1), 1)
        self.pool = nn.MaxPool2d(2)
        self.u4 = up_block(1024, 512)
        self.u3 = up_block(512, 256)
        self.u2 = up_block(256, 128)
        self.u1 = up_block(128, 64)

    def forward(self, x):

        x1 = self.c1(x)
        p1 = self.pool(x1)
        x2 = self.c2(p1)
        p2 = self.pool(x2)
        x3 = self.c3(p2)
        p3 = self.pool(x3)
        x4 = self.c4(p3)
        p4 = self.pool(x4)
        x5 = self.c5(p4)
        u4 = self.u4(x5)
        rc4 = copy_and_crop(u4.shape[2])(x4)
        c4 = torch.cat((rc4, u4), dim = 1)
        x6 = self.c6(c4)
        u3 = self.u3(x6)
        rc3 = copy_and_crop(u3.shape[2])(x3)
        c3 = torch.cat((rc3, u3), dim = 1)
        x7 = self.c7(c3)
        u2 = self.u2(x7)
        rc2 = copy_and_crop(u2.shape[2])(x2)
        c2 = torch.cat((rc2, u2), dim = 1)
        x8 = self.c8(c2)
        u1 = self.u1(x8)
        rc1 = copy_and_crop(u1.shape[2])(x1)
        c1 = torch.cat((rc1, u1),dim = 1)
        out_beta = self.c9(c1)
        output = self.out(out_beta)
        return output

def test():
    rand_data = torch.rand(1, 3, 572, 572)
    model = UNet(3,3)

    assert model(rand_data).shape == (1, 3, 388, 388), "Model Error"
    print(summary(model,(3,572,572), device = 'cpu'))

if __name__ == '__main__':
    test()