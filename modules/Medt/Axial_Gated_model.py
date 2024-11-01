import pdb
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import *
import pdb
import matplotlib.pyplot as plt
import random
from .axial_parts import*

class ResAxialAttentionUNet(nn.Module):
    
    def __init__(self, block, layers, zero_init_residual=True,
                 groups=8, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, s=0.125, img_size = 256,n_channels = 3,n_classes=1, bilinear=True):
        super(ResAxialAttentionUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes=n_classes
        # self.img_size=img_size
        self.bilinear = bilinear
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = int(64 * s)
        self.dilation = 1 #膨胀率
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(n_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.conv2 = nn.Conv2d(self.inplanes, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.bn2 = norm_layer(128)
        self.bn3 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, int(128 * s), layers[0], kernel_size= (img_size//2))
        self.layer2 = self._make_layer(block, int(256 * s), layers[1], stride=2, kernel_size=(img_size//2),
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, int(512 * s), layers[2], stride=2, kernel_size=(img_size//4),
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, int(1024 * s), layers[3], stride=2, kernel_size=(img_size//8),
                                       dilate=replace_stride_with_dilation[2])
        
        # Decoder
        self.decoder1 = nn.Conv2d(int(1024 *2*s)      ,        int(1024*2*s), kernel_size=3, stride=2, padding=1)
        self.decoder2 = nn.Conv2d(int(1024  *2*s)     , int(1024*s), kernel_size=3, stride=1, padding=1)
        self.decoder3 = nn.Conv2d(int(1024*s),  int(512*s), kernel_size=3, stride=1, padding=1)
        self.decoder4 = nn.Conv2d(int(512*s) ,  int(256*s), kernel_size=3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(int(256*s) , int(128*s) , kernel_size=3, stride=1, padding=1)
        self.adjust   = nn.Conv2d(int(128*s) , n_classes, kernel_size=1, stride=1, padding=0)
        self.soft     = nn.Softmax(dim=1)


    def _make_layer(self, block, planes, blocks, kernel_size=56, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation, 
                            norm_layer=norm_layer, kernel_size=kernel_size))
        self.inplanes = planes * block.expansion
        if stride != 1:
            kernel_size = kernel_size // 2

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, kernel_size=kernel_size))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        
        # AxialAttention Encoder
        # pdb.set_trace()

        #ConvBlock:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

    #EncoderBlock:
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        # print(x2.shape)
        x3 = self.layer3(x2)
        # print(x3.shape)
        x4 = self.layer4(x3)

    #DecoderBlock:(Conv+线性插值为2倍+skip跳连)
        x = F.relu(F.interpolate(self.decoder1(x4), scale_factor=(2,2), mode ='bilinear'))
        x = torch.add(x, x4)
        x = F.relu(F.interpolate(self.decoder2(x) , scale_factor=(2,2), mode ='bilinear'))
        x = torch.add(x, x3)
        x = F.relu(F.interpolate(self.decoder3(x) , scale_factor=(2,2), mode ='bilinear'))
        x = torch.add(x, x2)
        x = F.relu(F.interpolate(self.decoder4(x) , scale_factor=(2,2), mode ='bilinear'))
        x = torch.add(x, x1)
        x = F.relu(F.interpolate(self.decoder5(x) , scale_factor=(2,2), mode ='bilinear'))
        x = self.adjust(F.relu(x))
        # pdb.set_trace()
        return x

    def forward(self, x):
        return self._forward_impl(x)
