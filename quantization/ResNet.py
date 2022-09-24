import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

import math
import sys
sys.path.append("..")
from .quant import * 
from .quant_layer import * 
from .quan import *

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                    padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """
        Basic fp32 block of resnet

        Args:
            inplanes   (_type_)          : 
            planes     (_type_)          : 
            stride     (int, optional)   : . Defaults to 1.
            downsample (_type_, optional): . Defaults to None.
        """
        super(BasicBlock, self).__init__()
        self.conv1      = conv3x3(inplanes, planes, stride)
        self.bn1        = nn.BatchNorm2d(planes)
        self.relu       = nn.ReLU(inplace=True)
        self.conv2      = conv3x3(planes, planes)
        self.bn2        = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride     = stride

    def forward(self, x):
        quan_width=16
        x = FakeQuantize.apply(x, quan_width)
        
        residual = x

        out = self.conv1(x)
        out = FakeQuantize.apply(out, quan_width)
        out = self.bn1(out)
        out = FakeQuantize.apply(out, quan_width)
        out = self.relu(out)

        out = self.conv2(out)
        out = FakeQuantize.apply(out, quan_width)
        out = self.bn2(out)
        out = FakeQuantize.apply(out, quan_width)
        if self.downsample is not None:
            residual = self.downsample(x)
            residual = FakeQuantize.apply(residual, quan_width)
        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, fake_downsample=None):
        """
        Quantized basic block 

        Args:
            inplanes        (_type_)          : 
            planes          (_type_)          : 
            stride          (int, optional)   : . Defaults to 1.
            downsample      (_type_, optional): . Defaults to None.
            fake_downsample (_type_, optional): . Defaults to None.
        """
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * 4)

        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        quan_width = 16
        # ***** residual buffer ***** #
        residual = x
        # ***** 1st conv_bn layer ***** #
        out                      = self.conv1(x)        
        out                      = self.bn1(out)
        #conv1_weight, conv1_bias = quant_layer.ConvBNTrans(self.conv1, self.bn1)
        conv1_weight, conv1_bias = ConvBNTrans(self.conv1, self.bn1)
        #q_conv1_w,     q_conv1_b = quant.WeightQuant(conv1_weight, conv1_bias, 'conv')
        #q_conv1_w,     q_conv1_b = WeightQuant(conv1_weight, conv1_bias, 'conv')
        q_conv1_w = FakeQuantize.apply(conv1_weight, quan_width)
        q_conv1_b = FakeQuantize.apply(conv1_bias, quan_width)

        out                      = F.conv2d(x, q_conv1_w, q_conv1_b)
        #out                      = quant.FixedActQuant(out, 'conv')
        #out                      = FixedActQuant(out, 'conv')
        
        out                      = self.relu(out)
        out = FakeQuantize.apply(out, quan_width)
        # ***** 2nd conv_bn layer ***** #
        out_2                    = out
        out                      = self.conv2(out)
        out                      = self.bn2(out)
        #conv2_weight, conv2_bias = quant_layer.ConvBNTrans(self.conv2, self.bn2)
        conv2_weight, conv2_bias = ConvBNTrans(self.conv2, self.bn2)
        #q_conv2_w   , q_conv2_b  = quant.WeightQuant(conv2_weight, conv2_bias, 'conv')
        #q_conv2_w   , q_conv2_b  = WeightQuant(conv2_weight, conv2_bias, 'conv')
        q_conv2_w = FakeQuantize.apply(conv2_weight, quan_width)
        q_conv2_b = FakeQuantize.apply(conv2_bias, quan_width)

        out                      = F.conv2d(out_2, q_conv2_w, q_conv2_b, stride=self.stride, padding=1)        
        #out                      = quant.FixedActQuant(out, 'conv')
        #out                      = FixedActQuant(out, 'conv')
        
        out                      = self.relu(out)
        out = FakeQuantize.apply(out, quan_width)
        #print("每层的降采样卷积")
        #print(out.size())
        # ***** 3rd conv_bn layer ***** #
        out_3                    = out
        out                      = self.conv3(out)
        out                      = self.bn3(out)
        #conv3_weight, conv3_bias = quant_layer.ConvBNTrans(self.conv3, self.bn3)
        conv3_weight, conv3_bias = ConvBNTrans(self.conv3, self.bn3)
        #q_conv3_w   , q_conv3_b  = quant.WeightQuant(conv3_weight, conv3_bias, 'conv')
        #q_conv3_w   , q_conv3_b  = WeightQuant(conv3_weight, conv3_bias, 'conv')
        q_conv3_w = FakeQuantize.apply(conv3_weight, quan_width)
        q_conv3_b = FakeQuantize.apply(conv3_bias, quan_width)

        out                      = F.conv2d(out_3, q_conv3_w, q_conv3_b)        
        #out                      = quant.FixedActQuant(out, 'conv')
        #out                      = FixedActQuant(out, 'conv')
        out = FakeQuantize.apply(out, quan_width)
        # ***** downsampling config ***** #
        if self.downsample is not None:
            residual = self.downsample(x)
            #conv4_weight, conv4_bias = quant_layer.ConvBNTrans(self.downsample[0], self.downsample[1])
            conv4_weight, conv4_bias = ConvBNTrans(self.downsample[0], self.downsample[1])
            #conv4_weight, conv4_bias = quant.WeightQuant(conv4_weight, conv4_bias, 'conv')
            #conv4_weight, conv4_bias = WeightQuant(conv4_weight, conv4_bias, 'conv')
            q_conv4_w = FakeQuantize.apply(conv4_weight, quan_width)
            q_conv4_b = FakeQuantize.apply(conv4_bias, quan_width)
            residual = F.conv2d(x, q_conv4_w, q_conv4_b, stride=self.downsample[0].stride)
            #residual = quant.FixedActQuant(residual, 'conv')
            #residual = FixedActQuant(residual, 'conv')
            residual = FakeQuantize.apply(residual, quan_width)
        # ***** residual implementation ***** #
        out += residual
        out = self.relu(out)
        #out = quant.FixedActQuant(out, 'conv')
        #out = FixedActQuant(out, 'conv')
        out = FakeQuantize.apply(out, quan_width)
        return out



class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        #self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.RealWeight = torch.tensor([1,2,3,4])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        fake_downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
            
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, fake_downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
            
    def forward(self, x):
        quan_width = 16
        #x                        = quant.FixedActQuant(x, 'conv')
        #x                        = FixedActQuant(x, 'conv')
        x = FakeQuantize.apply(x, quan_width)
        #print("输入")
        #print(x.size())
        # ***** Residual preprocessing ***** #
        out                      = self.conv1(x)
        #print("7卷积")
        #print(out.size())
        out = FakeQuantize.apply(out, quan_width)
        out                      = self.bn1(out)
        out = FakeQuantize.apply(out, quan_width)
        #conv1_weight, conv1_bias = quant_layer.ConvBNTrans(self.conv1, self.bn1)
        conv1_weight, conv1_bias = ConvBNTrans(self.conv1, self.bn1)
        #q_conv1_w   , q_conv1_b  = quant.WeightQuant(conv1_weight, conv1_bias, 'conv')
        #q_conv1_w   , q_conv1_b  = WeightQuant(conv1_weight, conv1_bias, 'conv')
        q_conv1_w = FakeQuantize.apply(conv1_weight, quan_width)
        q_conv1_b = FakeQuantize.apply(conv1_bias, quan_width)
        out                      = F.conv2d(x, q_conv1_w, q_conv1_b, stride=2, padding=3)  
        #out                      = quant.FixedActQuant(out, 'conv')
        #out                      = FixedActQuant(out, 'conv')
        out                      = self.relu(out)
        out                      = self.maxpool(out)
        #print("池化")
        #print(out.size())
        out = FakeQuantize.apply(out, quan_width)
        # ***** Residual blocks ***** #
        out = self.layer1(out)
        print("第一层")
        print(out.size())
        out = self.layer2(out)
        print("第二层")
        print(out.size())
        out = self.layer3(out)
        print("第3层出来")
        print(out.size())
        out = self.layer4(out)
        # ***** Average pooling ***** #
        #out = self.avgpool(out)
        #out = out.view(x.size(0), -1)
        # ***** FC layer ***** #
        #out_fc = out
        #out    = self.fc(out)
        #out    = quant_layer.quant_fc(out_fc, self.fc)
        #out    = quant_fc(out_fc, self.fc)
        #print("最后输出")
        #print(out.size())
        return out


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model

def resnet50_2(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck_2, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model

def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
