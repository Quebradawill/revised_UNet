""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *

import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3,7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2,1,kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class Res_CBAM_DoubleConv(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out  # 广播机制
        out = self.sa(out) * out  # 广播机制
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
        
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
        #print(self.ratios)
        return self.maxpool_conv(x)

class CBAM_ASPP_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True,ratios=[3,6,9],gn=32,pool=False):
        super(CBAM_ASPP_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.ratios=ratios
        self.gn=gn
        self.pool=pool

        self.inc = DoubleConv(n_channels, 64,gn=gn)
        self.aspp1 = ASPP(64,64,self.ratios)
        self.ca1 = ChannelAttention(64)
        self.sa1 = SpatialAttention()
        self.down1 = Down(64, 128,gn=gn)
        self.aspp2 = ASPP(128,128,self.ratios)
        self.ca2 = ChannelAttention(128)
        self.sa2 = SpatialAttention()
        self.down2 = Down(128, 256,gn=gn)
        self.aspp3 = ASPP(256,256,self.ratios)
        self.ca3 = ChannelAttention(256)
        self.sa3 = SpatialAttention()
        self.down3 = Down(256, 512,gn=gn)
        self.aspp4 = ASPP(512,512,self.ratios)
        self.ca4 = ChannelAttention(512)
        self.sa4 = SpatialAttention()
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
        # x1 = self.ca1(x1) * x1  # 广播机制
        # x1 = self.sa1(x1) * x1
        # x1=self.se1(x1)
        # print("x1:{}".format(x1.shape))
        x2 = self.down1(x1)
        # x2 = self.ca2(x2) * x2  # 广播机制
        # x2 = self.sa2(x2) * x2 
        # x2=self.se2(x2)
        # print("x2:{}".format(x2.shape))
        x3 = self.down2(x2)
        # x3 = self.ca3(x3) * x3  # 广播机制
        # x3 = self.sa3(x3) * x3
        # x3=self.se3(x3)
        # print("x3:{}".format(x3.shape))
        x4 = self.down3(x3)
        # x4 = self.ca4(x4) * x4  # 广播机制
        # x4 = self.sa4(x4) * x4
        # x4=self.se4(x4)
        # print("x4:{}".format(x4.shape))
        
        x5 = self.down4(x4)
        # print("x5:{}".format(x5.shape))
        x4 = self.ca4(x4) * x4  # 广播机制
        x4 = self.sa4(x4) * x4
        # x4=self.aspp4(x4)
        # print("cbamx4:{}".format(x4.shape))
        x = self.up1(x5, x4)
        # print("x6:{}".format(x.shape))
        x3 = self.ca3(x3) * x3  # 广播机制
        x3 = self.sa3(x3) * x3
        # print("cbamx3:{}".format(x3.shape))
        # x3=self.aspp3(x3)
        x = self.up2(x, x3)
        # print("x7:{}".format(x.shape))
        x2 = self.ca2(x2) * x2  # 广播机制
        x2 = self.sa2(x2) * x2 
        # print("cbamx2:{}".format(x2.shape))
        # x2=self.aspp2(x2)
        x = self.up3(x, x2)
        # print("x8:{}".format(x.shape))
        x1 = self.ca1(x1) * x1  # 广播机制
        x1 = self.sa1(x1) * x1
        # print("cbamx1:{}".format(x1.shape))
        # x1=self.aspp1(x1)
        x = self.up4(x, x1)
        # print("x9:{}".format(x.shape))
        logits = self.outc(x)
        # print("out:{}".format(logits.shape))
        return logits
