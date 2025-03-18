import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .transformer import TransformerModule

def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ConvBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, previous_dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=previous_dilation, dilation=previous_dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class TransConvBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, previous_dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(TransConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=previous_dilation, dilation=previous_dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.attention_sptial = TransformerModule(in_channels=planes)
        self.attention_channel = ChannelAttention(planes)
        self.downsample = downsample
        self.stride = stride
        #self.gamma = 0

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        
        outa = self.attention_sptial(residual)
        #out = self.gamma*outa + out
        out = outa + out
        out += residual
        out = self.attention_channel(out)
        out = self.relu(out)
        return out


class SpatialConvBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, previous_dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SpatialConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=previous_dilation, dilation=previous_dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.attention_sptial = SpatialAttention()
        self.attention_channel = ChannelAttention(planes)
        self.downsample = downsample
        self.stride = stride
        #self.gamma = 0

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        
        outa = self.attention_sptial(residual)
        #out = self.gamma*outa + out
        out = outa + out
        out += residual
        out = self.attention_channel(out)
        out = self.relu(out)
        return out

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

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, norm_layer=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        inter_planes = inplanes/4
        self.conv1 = conv1x1(inplanes,inter_planes,stride=stride, dilation=dilation)
        self.bn1 = norm_layer(inter_planes)
        self.conv2 = conv3x3(inter_planes, inter_planes,stride=stride, dilation=dilation)
        self.bn2 = norm_layer(inter_planes)
        self.conv3 = conv1x1(inter_planes, planes * self.expansion,stride=stride, dilation=dilation)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.attention_sptial = SpatialAttention()
        self.attention_channel = ChannelAttention(planes)
        self.gamma = 0

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
            
        outa = self.attention_sptial(identity)

        out = self.gamma*outa + out

        out += identity
        out = self.attention_channel(out)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, blockA, blockB, layers,zero_init_residual=False, replace_stride_with_dilation=None,norm_layer=nn.BatchNorm2d):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.dilation = 1  #这里传入的dilation会影响反射填充时的padding值，影响特征图尺寸
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(blockA, 64, layers[0])
        self.layer2 = self._make_layer(blockA, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(blockB, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(blockB, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        ##初始化模型参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks,stride=1, dilate=False):
        norm_layer = nn.BatchNorm2d 
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
        layers.append(block(self.inplanes, planes, stride, downsample=downsample, 
                            dilation=previous_dilation, norm_layer=norm_layer))
        
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,stride=1, downsample=None,
                                dilation=self.dilation,norm_layer=norm_layer))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        c0 = self.layer0(x)
        c1 = self.layer1(c0)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return c1,c2,c3,c4
    
def _resnet(blockA,blockB, layers, **kwargs):
    model = ResNet(blockA,blockB, layers, **kwargs)
    return model

def ACResNet(layers=[2,2,2,2], **kwargs):
    return _resnet(ConvBlock, SpatialConvBlock, layers, **kwargs)


