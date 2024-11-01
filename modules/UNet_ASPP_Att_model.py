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

class Att_Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.att = Attention_block(F_g=in_channels // 2,F_l=in_channels // 2,F_int=out_channels//2)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.att = Attention_block(F_g=in_channels // 2,F_l=in_channels // 2,F_int=out_channels//2)
            self.conv = DoubleConv(in_channels, out_channels)
            #https://blog.csdn.net/qq_41076797/article/details/114494990


    def forward(self, x1, x2):#x2 is the output of down #huan
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

       #扩充维度   为什么不是裁剪x2？？？
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])  
        x2=self.att(g=x2,x=x1)####？？？？？谁是g,谁是x???
        x = torch.cat((x2, x1), dim=1)  
        
        return self.conv(x)


class ASPP_Att_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True,ratios=[3,6,9],gn=32,pool=False):
        super(ASPP_Att_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.ratios=ratios
        self.gn=gn
        self.pool=pool

        self.inc = DoubleConv(n_channels, 64)
        self.aspp1 = ASPP(64,64,self.ratios)
        self.down1 = Down(64, 128)
        self.aspp2 = ASPP(128,128,self.ratios)
        self.down2 = Down(128, 256)
        self.aspp3 = ASPP(256,256,self.ratios)
        self.down3 = Down(256, 512)
        self.aspp4 = ASPP(512,512,self.ratios)
        factor = 2 if bilinear else 1
        self.down4 = ASPP_Down(512, 1024 // factor)
        self.up1 = Att_Up(1024, 512 // factor, bilinear)
        self.up2 = Att_Up(512, 256 // factor, bilinear)
        self.up3 =Att_Up(256, 128 // factor, bilinear)
        self.up4 = Att_Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        # print("x1:{}".format(x1.shape))
        x2 = self.down1(x1)
        # print("x2:{}".format(x2.shape))
        x3 = self.down2(x2)
        # print("x3:{}".format(x3.shape))
        x4 = self.down3(x3)
        # print("x4:{}".format(x4.shape))
        x5 = self.down4(x4)

        # x4=self.aspp4(x4)
        # print("asppx4:{}".format(x4.shape))
        x = self.up1(x5, x4)
        # print("x4x5:{}".format(x.shape))
        # x3=self.aspp3(x3)
        # print("asppx3:{}".format(x3.shape))             
        x = self.up2(x, x3)
        # x2=self.aspp2(x2)
        # print("asppx2:{}".format(x2.shape))
        x = self.up3(x, x2)
        # x1=self.aspp1(x1)
        # print("asppx1:{}".format(x1.shape))
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
