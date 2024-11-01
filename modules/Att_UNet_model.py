""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *



class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        # print("g:{}".format(g.shape))
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class Att_Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.att = Attention_block(F_g=in_channels // 2,F_l=in_channels // 2,F_int=out_channels//2)
            self.conv = BNDoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.att = Attention_block(F_g=in_channels // 2,F_l=in_channels // 2,F_int=out_channels//2)
            self.conv = BNDoubleConv(in_channels, out_channels)
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



class Att_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(Att_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = BNDoubleConv(n_channels,64)
        self.down1 = BNDown(64, 128)
        self.down2 = BNDown(128, 256)
        self.down3 = BNDown(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = BNDown(512, 1024 // factor)
        self.up1 = Att_Up(1024, 512 // factor, bilinear)
        self.up2 = Att_Up(512, 256 // factor, bilinear)
        self.up3 = Att_Up(256, 128 // factor, bilinear)
        self.up4 = Att_Up(128, 64, bilinear)
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
