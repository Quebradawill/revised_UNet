""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *


class Logo_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(Logo_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x0=x.clone()
        
        #global
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        #local
        x_loc=x.clone()#局部分割结果图
        for i in range(0,4):
            for j in range(0,4):
                x_l=x0[:,:,64*i:64*(i+1),64*j:64*(j+1)]   
                x_l1 = self.inc(x_l)
                x_l2= self.down1(x_l1)
                x_l3= self.down2(x_l2)
                x_l4 = self.down3(x_l3)
                x_l5 = self.down4(x_l4)
                x_l= self.up1(x_l5, x_l4)
                x_l = self.up2(x_l, x_l3)
                x_l = self.up3(x_l, x_l2)
                x_l = self.up4(x_l, x_l1)
                x_loc[:,:,64*i:64*(i+1),64*j:64*(j+1)] = x_l

        x=torch.add(x,x_loc)

        logits = self.outc(x)
        return logits
