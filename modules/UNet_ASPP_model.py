""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *
class ASPP_Down(nn.Module):
    """Downscaling with maxpool then double conv and aspp"""

    def __init__(self, in_channels, out_channels, gn=4,ratios=[3,6,9],pool=False):
        super().__init__()
        self.gn=gn
        self.ratios=ratios
        self.pool=pool
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ASPP(in_channels,out_channels,self.ratios,self.pool),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            torch.nn.GroupNorm(gn, out_channels, eps=1e-05, affine=True),
            nn.ReLU(inplace=True),
            ASPP(out_channels,out_channels,self.ratios,self.pool),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            torch.nn.GroupNorm(gn, out_channels, eps=1e-05, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # print(self.pool)
        return self.maxpool_conv(x)
class ASPP_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True,ratios=[3,6,9],gn=32,pool=False):
        super(ASPP_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.ratios=ratios
        self.gn=gn
        self.pool=pool

        self.inc = DoubleConv(n_channels, 64,gn=gn)
        self.aspp1 = ASPP(64,64,self.ratios)
        self.down1 = Down(64, 128,gn=gn)
        self.aspp2 = ASPP(128,128,self.ratios)
        self.down2 = Down(128, 256,gn=gn)
        self.aspp3 = ASPP(256,256,self.ratios)

        self.down3 = Down(256, 512,gn=gn)
        self.aspp4 = ASPP(512,512,self.ratios)
        factor = 2 if bilinear else 1
        self.down4 = ASPP_Down(512, 1024 // factor,gn=gn,ratios=ratios,pool=pool)
        # self.down4 = Down(512, 1024 // factor,gn=gn)
        self.up1 = Up(1024, 512 // factor, bilinear,gn=gn)
        self.up2 = Up(512, 256 // factor, bilinear,gn=gn)
        self.up3 = Up(256, 128 // factor, bilinear,gn=gn)
        self.up4 = Up(128, 64, bilinear,gn=gn)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)     
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits
