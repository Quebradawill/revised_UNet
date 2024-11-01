""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *

#引入残差连接的U-Net网络

class ResUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True,gn=32):
        super(ResUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.gn=gn

        if gn==0:
            self.inc = ResBNDoubleConv(n_channels, 64,gn=gn)
            self.down1 = ResBNDown(64, 128,gn=gn)
            self.down2 = ResBNDown(128, 256,gn=gn)
            self.down3 = ResBNDown(256, 512,gn=gn)
            factor = 2 if bilinear else 1
            self.down4 = ResBNDown(512, 1024 // factor,gn=gn)
            self.up1 = ResBNUp(1024, 512 // factor, bilinear,gn=gn)
            self.up2 = ResBNUp(512, 256 // factor, bilinear,gn=gn)
            self.up3 = ResBNUp(256, 128 // factor, bilinear,gn=gn)
            self.up4 = ResBNUp(128, 64, bilinear,gn=gn)
            self.outc = OutConv(64, n_classes)
        else:
            self.inc = ResDoubleConv(n_channels, 64,gn=gn)
            self.down1 = ResDown(64, 128,gn=gn)
            self.down2 = ResDown(128, 256,gn=gn)
            self.down3 = ResDown(256, 512,gn=gn)
            factor = 2 if bilinear else 1
            self.down4 = ResDown(512, 1024 // factor,gn=gn)
            self.up1 = ResUp(1024, 512 // factor, bilinear,gn=gn)
            self.up2 = ResUp(512, 256 // factor, bilinear,gn=gn)
            self.up3 = ResUp(256, 128 // factor, bilinear,gn=gn)
            self.up4 = ResUp(128, 64, bilinear,gn=gn)
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
