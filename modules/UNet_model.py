""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True,gn=32):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.gn=gn

        # self.inc = DoubleConv(n_channels, 64,gn=gn)
        # self.down1 = Down(64, 128,gn=gn)
        # self.down2 = Down(128, 256,gn=gn)
        # self.down3 = Down(256, 512,gn=gn)
        # factor = 2 if bilinear else 1
        # self.down4 = Down(512, 1024 // factor,gn=gn)
        # self.up1 = Up(1024, 512 // factor, bilinear,gn=gn)
        # self.up2 = Up(512, 256 // factor, bilinear,gn=gn)
        # self.up3 = Up(256, 128 // factor, bilinear,gn=gn)
        # self.up4 = Up(128, 64, bilinear,gn=gn)
        # self.outc = OutConv(64, n_classes)

        self.inc = BNDoubleConv(n_channels, 64)
        self.down1 = BNDown(64, 128)
        self.down2 = BNDown(128, 256)
        self.down3 = BNDown(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = BNDown(512, 1024 // factor)
        self.up1 = BNUp(1024, 512 // factor, bilinear)
        self.up2 = BNUp(512, 256 // factor, bilinear)
        self.up3 = BNUp(256, 128 // factor, bilinear)
        self.up4 = BNUp(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # print("x4:{}".format(x4.shape))
        # print("x5:{}".format(x5.shape))
        x = self.up1(x5, x4)
        # print("x4x5:{}".format(x.shape))
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
