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


"""
ResNet34 + U-Net
"""
import torch
from torch import nn
import torchvision.models as models
import torch.nn.functional as F
from torchsummary import summary


class expansive_block(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(expansive_block, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=in_channels, out_channels=mid_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(kernel_size=(3, 3), in_channels=mid_channels, out_channels=out_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, d, e=None):
        d = F.interpolate(d, scale_factor=2, mode='bilinear', align_corners=True)
        # concat

        if e is not None:
            cat = torch.cat([e, d], dim=1)
            out = self.block(cat)
        else:
            out = self.block(d)
        return out


def final_block(in_channels, out_channels):
    block = nn.Sequential(
        nn.Conv2d(kernel_size=(3, 3), in_channels=in_channels, out_channels=out_channels, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
    )
    return block


class Resnet34_Unet(nn.Module):

    def __init__(self, in_channel, out_channel, pretrained=False):
        super(Resnet34_Unet, self).__init__()

        self.resnet = models.resnet34(pretrained=pretrained)
        self.layer0 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool
        )

        # Encode
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4

        # Bottleneck
        self.bottleneck = torch.nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=512, out_channels=1024, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(kernel_size=(3, 3), in_channels=1024, out_channels=1024, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )

        # Decode
        self.conv_decode4 = expansive_block(1024+512, 512, 512)
        self.conv_decode3 = expansive_block(512+256, 256, 256)
        self.conv_decode2 = expansive_block(256+128, 128, 128)
        self.conv_decode1 = expansive_block(128+64, 64, 64)
        self.conv_decode0 = expansive_block(64, 32, 32)
        # self.final_layer = final_block(32, out_channel)
        self.outc = OutConv(32, out_channel)

    def forward(self, x):
        x = self.layer0(x)
        # Encode
        encode_block1 = self.layer1(x)
        encode_block2 = self.layer2(encode_block1)
        encode_block3 = self.layer3(encode_block2)
        encode_block4 = self.layer4(encode_block3)

        # Bottleneck
        bottleneck = self.bottleneck(encode_block4)

        # Decode
        decode_block4 = self.conv_decode4(bottleneck, encode_block4)
        decode_block3 = self.conv_decode3(decode_block4, encode_block3)
        decode_block2 = self.conv_decode2(decode_block3, encode_block2)
        decode_block1 = self.conv_decode1(decode_block2, encode_block1)
        decode_block0 = self.conv_decode0(decode_block1)

        # final_layer = self.final_layer(decode_block0)
        final_layer= self.outc(decode_block0)
        

        return final_layer


# flag = 0

# if flag:
#     image = torch.rand(1, 3, 572, 572)
#     Resnet34_Unet = Resnet34_Unet(in_channel=3, out_channel=1)
#     mask = Resnet34_Unet(image)
#     print(mask.shape)

# # 测试网络
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Resnet34_Unet(in_channel=1, out_channel=1, pretrained=True).to(device)
# summary(model, input_size=(3, 512, 512))





# class Res_UNet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=True,reduce=16,gn=32):
#         super(Res_UNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear
#         self.reduce=reduce
#         self.gn=gn

#         self.inc = DoubleConv(n_channels, 64,gn=gn)
#         self.se1 =SE_block(64)
#         self.down1 = Down(64, 128,gn=gn)
#         self.se2 =SE_block(128)
#         self.down2 = Down(128, 256,gn=gn)
#         self.se3 =SE_block(256)
#         self.down3 = Down(256, 512,gn=gn)
#         self.se4 =SE_block(512)
#         factor = 2 if bilinear else 1
#         self.down4 = Down(512, 1024 // factor,gn=gn)
#         self.up1 = Up(1024, 512 // factor, bilinear,gn=gn)
#         self.up2 = Up(512, 256 // factor, bilinear,gn=gn)
#         self.up3 = Up(256, 128 // factor, bilinear,gn=gn)
#         self.up4 = Up(128, 64, bilinear,gn=gn)
#         self.outc = OutConv(64, n_classes)

#     def forward(self, x):
#         x1 = self.inc(x)
#         # x1=self.se1(x1)
#         # print("x1:{}".format(x1.shape))
#         x2 = self.down1(x1)
#         # x2=self.se2(x2)
#         # print("x2:{}".format(x2.shape))
#         x3 = self.down2(x2)
#         # x3=self.se3(x3)
#         # print("x3:{}".format(x3.shape))
#         x4 = self.down3(x3)
#         # x4=self.se4(x4)
#         # print("x4:{}".format(x4.shape))
        
#         x5 = self.down4(x4)
#         x4=self.se4(x4)
#         x = self.up1(x5, x4)
#         x3=self.se3(x3)
#         x = self.up2(x, x3)
#         x2=self.se2(x2)       
#         x = self.up3(x, x2)
#         x1=self.se1(x1)
#         x = self.up4(x, x1)
#         logits = self.outc(x)
#         return logits



