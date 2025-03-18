import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import models
from .backbone import ACResNet

def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()

class ResNet(nn.Module):
    def __init__(self, in_channels=3, backbone='resnet18', pretrained=False):
        super(ResNet, self).__init__()
        model = getattr(models, backbone)(pretrained)
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        c1 = x
        x = self.layer2(x)
        c2 = x
        x = self.layer3(x)
        c3 = x
        x = self.layer4(x)
        c4 = x
        return c1,c2,c3,c4

class Decoder(nn.Module):
    def __init__(self, low_level_channels,high_level_channels):
        super(Decoder, self).__init__()
        self.output = nn.Sequential(
            nn.Conv2d(low_level_channels + high_level_channels, low_level_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(low_level_channels),
            nn.ReLU(inplace=True),
        )
        initialize_weights(self)

    def forward(self, low_level_features, x):
        H, W = low_level_features.size(2), low_level_features.size(3)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        x = self.output(torch.cat((low_level_features, x), dim=1))
        return x

class final_conv(nn.Module):
    def __init__(self,in_channels, num_classes):
        super(final_conv, self).__init__()
        self.output = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(in_channels, num_classes, 1))
        initialize_weights(self)

    def forward(self, x):
        out = self.output(x)
        return out

class Net(nn.Module):
    def __init__(self, num_classes,  freeze_bn=False, **_):
        super(Net, self).__init__()
        self.backbone = ACResNet()
        self.decoder4 = Decoder(512,1024)
        self.attention4 = DAM(512)
        self.decoder3 = Decoder(256,512)
        self.attention3 = DAM(256)
        self.decoder2 = Decoder(128,256)
        self.attention2 = DAM(128)
        self.final_conv = final_conv(128,num_classes)
        #initialize_weights(self)
        if freeze_bn: self.freeze_bn()

    def forward(self, xa, xb):
        H, W = xb.size(2), xb.size(3)
        c11,c21,c31,c41 = self.backbone(xa)
        c12,c22,c32,c42 = self.backbone(xb)
        c1 = torch.cat([c11,c12],dim=1)
        c2 = torch.cat([c21,c22],dim=1)
        c3 = torch.cat([c31,c32],dim=1)
        c4 = torch.cat([c41,c42],dim=1)

        x = self.decoder4(c3,c4)
        x = self.attention4(x)
        x = self.decoder3(c2,x)
        x = self.attention3(x)
        x = self.decoder2(c1,x)
        x = self.attention2(x)
        x = self.final_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return (x,)

class SE(nn.Module):
    def __init__(self, in_channels, ratio=4):
        super(SE, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        #卷积核，用多大尺寸？
        self.compress = nn.Conv2d(in_channels, in_channels // ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_channels // ratio, in_channels, 1, 1, 0)
        self.relu = nn.ReLU()
        self.sigmod = nn.Sigmoid()
    def forward(self, x):
        #仅使用平均池化压缩不同位置的信息
        out = self.squeeze(x)
        out = self.compress(out)
        out = self.relu(out)
        out = self.excitation(out)
        out = self.sigmod(out)
        return x*out
    
# 通道注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_chnls, ratio=4):
        super(ChannelAttention, self).__init__()
        self.squeeze_a = nn.AdaptiveAvgPool2d((1, 1))
        self.squeeze_m = nn.AdaptiveMaxPool2d((1,1))
        self.compress = nn.Conv2d(in_chnls, in_chnls // ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls // ratio, in_chnls, 1, 1, 0)
        self.relu = nn.ReLU()
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        #使用平均池化压缩不同位置的信息
        outa = self.squeeze_a(x)
        outa = self.compress(outa)
        outa = self.relu(outa)
        outa = self.excitation(outa)
        #使用最大池化压缩不同位置的信息
        outm = self.squeeze_m(x)
        outm = self.compress(outm)
        outm = self.relu(outm)
        outm = self.excitation(outm)
        out = outa + outm
        out = self.sigmod(out)
        return x*out

class Efficine_Channel_Attention(nn.Module):
    def __init__(self, channels, gamma = 2, b = 1):
        super(Efficine_Channel_Attention, self).__init__()
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size = kernel_size, padding = (kernel_size - 1) // 2, bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_x = self.avg_pooling(x)
        max_x = self.max_pooling(x)
        avg_out = self.conv(avg_x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        max_out = self.conv(max_x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        v = self.sigmoid(avg_out + max_out)
        return x * v

# 空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #X[BCHW]
        #计算张量在维度1（通道方向）上的平均值
        avg_out = torch.mean(x, dim=1, keepdim=True)
        #计算张量在维度1（通道方向）上的最大值
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        ##同一位置所有通道压缩成一个
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)
        return x*out

class DAM(nn.Module):
    def __init__(self, in_channels, kernel_size=7):
        super(DAM, self).__init__()
        self.cam = Efficine_Channel_Attention(in_channels)
        self.sam = SpatialAttention(kernel_size)
        initialize_weights(self)

    def forward(self, x):
        x_cam = self.cam(x)
        x_sam = self.sam(x)
        output = x_cam + x_sam
        return output
