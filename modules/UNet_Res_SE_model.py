""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *

#引入残差连接and  SE block 的U-Net网络

class ResSeUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True,reduce=16,gn=32):
        super(ResSeUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.reduce=reduce
        self.gn=gn

        if gn==0:
            self.inc = ResBNDoubleConv(n_channels, 64,gn=gn)
            self.se1 =SE_block(64)
            self.down1 = ResBNDown(64, 128,gn=gn)
            self.se2 =SE_block(128)
            self.down2 = ResBNDown(128, 256,gn=gn)
            self.se3 =SE_block(256)
            self.down3 = ResBNDown(256, 512,gn=gn)
            self.se4 =SE_block(512)
            factor = 2 if bilinear else 1
            self.down4 = ResBNDown(512, 1024 // factor,gn=gn)
            self.up1 = ResBNUp(1024, 512 // factor, bilinear,gn=gn)
            self.up2 = ResBNUp(512, 256 // factor, bilinear,gn=gn)
            self.up3 = ResBNUp(256, 128 // factor, bilinear,gn=gn)
            self.up4 = ResBNUp(128, 64, bilinear,gn=gn)
            self.outc = OutConv(64, n_classes)           
        
        else:
            self.inc = ResDoubleConv(n_channels, 64,gn=gn)
            self.se1 =SE_block(64)
            self.down1 = ResDown(64, 128,gn=gn)
            self.se2 =SE_block(128)
            self.down2 = ResDown(128, 256,gn=gn)
            self.se3 =SE_block(256)
            self.down3 = ResDown(256, 512,gn=gn)
            self.se4 =SE_block(512)
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
        x4=self.se4(x4)
        x = self.up1(x5, x4)
        x3=self.se3(x3)
        x = self.up2(x, x3)
        x2=self.se2(x2)
        x = self.up3(x, x2)
        x1=self.se1(x1)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
