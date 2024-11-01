""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *

import torch
import torch.nn as nn

# 定义residual
class RB(nn.Module):
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1):
        super(RB, self).__init__()
        self.rb = nn.Sequential(nn.Conv2d(nin, nout, ksize, stride, pad),
                                nn.BatchNorm2d(nout),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(nin, nout, ksize, stride, pad),
                                nn.BatchNorm2d(nout))
    def forward(self, input):
        x = input
        x = self.rb(x)
        return nn.ReLU(input + x)

# 定义SE模块
class SE(nn.Module):
    def __init__(self, nin, nout, reduce=16):
        super(SE, self).__init__()
        self.gp = nn.AvgPool2d(1)
        self.rb1 = RB(nin, nout)
        self.se = nn.Sequential(nn.Linear(nout, nout // reduce),
                                nn.ReLU(inplace=True),
                                nn.Linear(nout // reduce, nout),
                                nn.Sigmoid())
    def forward(self, input):
        x = input
        x = self.rb1(x)

        b, c, _, _ = x.size()
        y = self.gp(x).view(b, c)
        y = self.se(y).view(b, c, 1, 1)
        y = x * y.expand_as(x)
        # out = y + input
        return out

class SE_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True,reduce=16,gn=32):
        super(SE_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.reduce=reduce
        self.gn=gn

        if gn==0:
            # print('BN')
            self.inc = BNDoubleConv(n_channels, 64)
            self.se1 =SE_block(64)
            self.down1 = BNDown(64, 128)
            self.se2 =SE_block(128)
            self.down2 = BNDown(128, 256)
            self.se3 =SE_block(256)
            self.down3 = BNDown(256, 512)
            self.se4 =SE_block(512)
            factor = 2 if bilinear else 1
            self.down4 = BNDown(512, 1024 // factor)
            self.up1 = BNUp(1024, 512 // factor, bilinear)
            self.up2 = BNUp(512, 256 // factor, bilinear)
            self.up3 = BNUp(256, 128 // factor, bilinear)
            self.up4 = BNUp(128, 64, bilinear)
            self.outc = OutConv(64, n_classes)
        else:
            self.inc = DoubleConv(n_channels, 64,gn=gn)
            self.se1 =SE_block(64)
            self.down1 = Down(64, 128,gn=gn)
            self.se2 =SE_block(128)
            self.down2 = Down(128, 256,gn=gn)
            self.se3 =SE_block(256)
            self.down3 = Down(256, 512,gn=gn)
            self.se4 =SE_block(512)
            factor = 2 if bilinear else 1
            self.down4 = Down(512, 1024 // factor,gn=gn)
            self.up1 = Up(1024, 512 // factor, bilinear,gn=gn)
            self.up2 = Up(512, 256 // factor, bilinear,gn=gn)
            self.up3 = Up(256, 128 // factor, bilinear,gn=gn)
            self.up4 = Up(128, 64, bilinear,gn=gn)
            self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # print(self.gn)
        x1 = self.inc(x)
        # x1=self.se1(x1)
        # print("x1:{}".format(x1.shape))
        x2 = self.down1(x1)
        # x2=self.se2(x2)
        # print("x2:{}".format(x2.shape))
        x3 = self.down2(x2)
        # x3=self.se3(x3)
        # print("x3:{}".format(x3.shape))
        x4 = self.down3(x3)
        # x4=self.se4(x4)
        # print("x4:{}".format(x4.shape))
        
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
