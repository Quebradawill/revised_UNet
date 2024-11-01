""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

class BNDoubleConv(nn.Module):
    """(convolution => [GN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            # torch.nn.GroupNorm(gn, mid_channels, eps=1e-05, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            # torch.nn.GroupNorm(gn, out_channels, eps=1e-05, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # print("gn:{}".format(self.gn))
        return self.double_conv(x)
class BNDown(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            BNDoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class BNUp(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = BNDoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = BNDoubleConv(in_channels, out_channels)
            #https://blog.csdn.net/qq_41076797/article/details/114494990

    def forward(self, x1, x2):#x2 is the output of down #huan
        x1 = self.up(x1)
        # print("upx:{}".format(x1.shape)) 
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

       #扩充维度   为什么不是裁剪x2？？？

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # print("reshape_upx:{}".format(x1.shape))
        x = torch.cat((x2, x1), dim=1) 
        # print("conx:{}".format(x.shape)) 
        return self.conv(x)     

class DoubleConv(nn.Module):
    """(convolution => [GN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None,gn=4):
        super().__init__()
        self.gn=gn
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(mid_channels),
            torch.nn.GroupNorm(gn, mid_channels, eps=1e-05, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            torch.nn.GroupNorm(gn, out_channels, eps=1e-05, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # print("gn:{}".format(self.gn))
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels,gn):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels,gn=gn)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear, gn):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2,gn=gn)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels,gn=gn)
            #https://blog.csdn.net/qq_41076797/article/details/114494990

    def forward(self, x1, x2):#x2 is the output of down #huan
        x1 = self.up(x1)
        # print("upx:{}".format(x1.shape)) 
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

       #扩充维度   为什么不是裁剪x2？？？

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # print("reshape_upx:{}".format(x1.shape))
        x = torch.cat((x2, x1), dim=1) 
        # print("conx:{}".format(x.shape)) 
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class ResDoubleConv(nn.Module):
    """(convolution => [GN] => ReLU) convolution => [GN] =+x=> ReLU"""

    def __init__(self, in_channels, out_channels, mid_channels=None,gn=4):
        super().__init__()
        self.gn=gn
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(mid_channels),
            torch.nn.GroupNorm(gn, mid_channels, eps=1e-05, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),       
            torch.nn.GroupNorm(gn, out_channels, eps=1e-05, affine=True)
        )


        self.shortcut = nn.Sequential()
        if in_channels != out_channels :
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                torch.nn.GroupNorm(gn, out_channels, eps=1e-05, affine=True)
            )

    def forward(self, x):
        # print("gn:{}".format(self.gn))
        return nn.ReLU(inplace=True)(self.double_conv(x)+self.shortcut(x))

class ResDown(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels,gn):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ResDoubleConv(in_channels, out_channels,gn=gn)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class ResUp(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear, gn):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ResDoubleConv(in_channels, out_channels, in_channels // 2,gn=gn)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = ResDoubleConv(in_channels, out_channels,gn=gn)
            #https://blog.csdn.net/qq_41076797/article/details/114494990

    def forward(self, x1, x2):#x2 is the output of down #huan
        x1 = self.up(x1)
        # print("upx:{}".format(x1.shape)) 
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

       #扩充维度   为什么不是裁剪x2？？？

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # print("reshape_upx:{}".format(x1.shape))
        x = torch.cat((x2, x1), dim=1) 
        # print("conx:{}".format(x.shape)) 
        return self.conv(x)

class ResBNDoubleConv(nn.Module):
    """(convolution => [GN] => ReLU) convolution => [GN] =+x=> ReLU"""

    def __init__(self, in_channels, out_channels, mid_channels=None,gn=4):
        super().__init__()
        self.gn=gn
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            # torch.nn.GroupNorm(gn, mid_channels, eps=1e-05, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            # torch.nn.GroupNorm(gn, out_channels, eps=1e-05, affine=True)
        )


        self.shortcut = nn.Sequential()
        if in_channels != out_channels :
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                # torch.nn.GroupNorm(gn, out_channels, eps=1e-05, affine=True)
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # print("gn:{}".format(self.gn))
        return nn.ReLU(inplace=True)(self.double_conv(x)+self.shortcut(x))

class ResBNDown(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels,gn):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ResBNDoubleConv(in_channels, out_channels,gn=gn)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class ResBNUp(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear, gn):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ResBNDoubleConv(in_channels, out_channels, in_channels // 2,gn=gn)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = ResBNDoubleConv(in_channels, out_channels,gn=gn)
            #https://blog.csdn.net/qq_41076797/article/details/114494990

    def forward(self, x1, x2):#x2 is the output of down #huan
        x1 = self.up(x1)
        # print("upx:{}".format(x1.shape)) 
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

       #扩充维度   为什么不是裁剪x2？？？

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # print("reshape_upx:{}".format(x1.shape))
        x = torch.cat((x2, x1), dim=1) 
        # print("conx:{}".format(x.shape)) 
        return self.conv(x)



class SE_block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
class Res_SE_block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Res_SE_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        x = input
        # x = self.rb1(x)
        b, c, _, _ = x.size()
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = x * y.expand_as(x)
        out = y + input
        return out
class Res_SE_DoubleConv(nn.Module):
    """(convolution => [GN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None,gn=4):
        super().__init__()
        self.gn=gn
        if not mid_channels:
            mid_channels = out_channels
        self.double1_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1,padding=1),
            # nn.BatchNorm2d(mid_channels),
            torch.nn.GroupNorm(gn, mid_channels, eps=1e-05, affine=True),
            nn.ReLU(inplace=True),
        )
        self.double2_conv = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3,stride=1, padding=1),
            # nn.BatchNorm2d(out_channels),
            torch.nn.GroupNorm(gn, out_channels, eps=1e-05, affine=True),
        )
        self.se=SE_block(out_channels)
        self.relu=nn.ReLU(inplace=True)

    def forward(self, x):
        # print("gn:{}".format(self.gn))
        x1=self.double1_conv(x)
        x2=self.double2_conv(x1)
        y=self.se(x1)
        out=self.relu(x2+y)
        return out
class Res_SE_Down(nn.Module):
    """Downscaling with maxpool then Res_SE_DoubleConv"""

    def __init__(self, in_channels, out_channels,gn):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            Res_SE_DoubleConv(in_channels, out_channels,gn=gn)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
class Res_SE_DoubleConv2(nn.Module):
    """(convolution => [GN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None,gn=4):
        super().__init__()
        self.gn=gn
        if not mid_channels:
            mid_channels = out_channels
        self.double1_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1,padding=1),
            # nn.BatchNorm2d(mid_channels),
            torch.nn.GroupNorm(gn, mid_channels, eps=1e-05, affine=True),
            nn.ReLU(inplace=True),
        )
        self.double2_conv = nn.Sequential(
            SE_block(out_channels),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3,stride=1, padding=1),
            # nn.BatchNorm2d(out_channels),
            torch.nn.GroupNorm(gn, out_channels, eps=1e-05, affine=True),
        )
        
        self.relu=nn.ReLU(inplace=True)

    def forward(self, x):
        # print("gn:{}".format(self.gn))
        x1=self.double1_conv(x)
        x2=self.double2_conv(x1)
        out=self.relu(x1+x2)
        return out
class Res_SE_Down2(nn.Module):
    """Downscaling with maxpool then Res_SE_DoubleConv"""

    def __init__(self, in_channels, out_channels,gn):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            Res_SE_DoubleConv2(in_channels, out_channels,gn=gn)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class ASPP(nn.Module):
    def __init__(self, in_channel=512, depth=256,ratios=[3,6,9],pool=False):
        super(ASPP,self).__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.ratios=ratios
        self.pool=pool
        # k=1 s=1 no pad
        #[1 2 3 5]/[1 6 12 18]
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=ratios[0], dilation=ratios[0])
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=ratios[1], dilation=ratios[1])
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=ratios[2], dilation=ratios[2])
        if self.pool=='True':
            self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)
        else:
            self.conv_1x1_output = nn.Conv2d(depth * 4, depth, 1, 1)
 
    def forward(self, x):
        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)
        # print("atrous_block1:{}".format(atrous_block1.shape))
        # print("atrous_block6:{}".format(atrous_block6.shape))
        # print("atrous_block12:{}".format(atrous_block12.shape))
        # print("atrous_block18:{}".format(atrous_block18.shape))
        # print("ratio{}".format(self.ratios))
        # print("aspp_poll:{}".format(self.pool))
        if self.pool=='True':
            # print("pool is True")
            # print("aspp_poll:{}".format(self.pool))
            size = x.shape[2:]
            # print("asppsize:{}".format(size))
            image_features = self.mean(x)#全局平均池化
            # print("image_features:{}".format(image_features.shape))
            image_features = self.conv(image_features)
            # print("convimage_features:{}".format(image_features.shape))
            # image_features = F.upsample(image_features, size=size, mode='bilinear')   
            image_features = F.interpolate(image_features, size=size, mode='bilinear',align_corners=True)     
            # print("image_features2:{}".format(image_features.shape))
            net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                                atrous_block12, atrous_block18], dim=1))
        else:
            # print("pool is False")
            # print("aspp_poll:{}".format(self.pool))
            net = self.conv_1x1_output(torch.cat([atrous_block1, atrous_block6,
                                                atrous_block12, atrous_block18], dim=1))
        return net

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
        x2=self.att(g=x1,x=x2)
        x = torch.cat((x2, x1), dim=1)  
        
        return self.conv(x)




class Recurrent_block(nn.Module):
    def __init__(self,ch_out,t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x1 = self.conv(x+x1)
        return x1
        
class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1

class R2_Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            RRCNN_block(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class R2_Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            #self.att = Attention_block(F_g=in_channels // 2,F_l=in_channels // 2,F_int=out_channels//2)
            self.conv = RRCNN_block(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            #self.att = Attention_block(F_g=in_channels // 2,F_l=in_channels // 2,F_int=out_channels//2)
            self.conv = RRCNN_block(in_channels, out_channels)

    def forward(self, x1, x2):#x2 is the output of down #huan
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])  
        x = torch.cat((x2, x1), dim=1)  
        
        return self.conv(x)

class Att_R2_Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.att = Attention_block(F_g=in_channels // 2,F_l=in_channels // 2,F_int=out_channels//2)
            self.conv = RRCNN_block(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.att = Attention_block(F_g=in_channels // 2,F_l=in_channels // 2,F_int=out_channels//2)
            self.conv = RRCNN_block(in_channels, out_channels)

    def forward(self, x1, x2):#x2 is the output of down #huan
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])  
        x2=self.att(g=x1,x=x2)
        x = torch.cat((x2, x1), dim=1)  
        
        return self.conv(x)   