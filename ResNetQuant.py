import torch
import torch.nn as nn
from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU, QuantMaxPool2d, QuantIdentity
from brevitas.quant import Int8WeightPerTensorFixedPoint
from torch.nn import BatchNorm2d

from common import CommonWeightQuant, CommonActQuant


class BottleneckQuant(nn.Module):
    def __init__(self, in_planes, planes, stride=1, weight_bit_width=1, act_bit_width=1):
        super(BottleneckQuant, self).__init__()

        weight_quant = CommonWeightQuant
        act_quant = CommonActQuant

        self.conv1 = QuantConv2d(in_planes, planes, kernel_size=1, bias=False, weight_quant=weight_quant,
                                 bit_width=weight_bit_width)
        self.bn1 = BatchNorm2d(planes)
        self.relu1 = QuantReLU(act_quant=act_quant, bit_width=act_bit_width),

        self.conv2 = QuantConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
                                 weight_quant=weight_quant, bit_width=weight_bit_width)
        self.bn2 = BatchNorm2d(planes)
        self.relu2 = QuantReLU(act_quant=act_quant, bit_width=act_bit_width),

        self.conv3 = QuantConv2d(planes, planes * 4, kernel_size=1, bias=False, weight_quant=weight_quant,
                                 bit_width=weight_bit_width)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu3 = QuantReLU(act_quant=act_quant, bit_width=act_bit_width),

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * 4:
            self.shortcut = nn.Sequential(
                QuantConv2d(in_planes, planes * 4, kernel_size=1, stride=stride, bias=False, weight_quant=weight_quant,
                            bit_width=weight_bit_width),
                BatchNorm2d(planes * 4)
            )

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu3(out)
        return out


class ResNet50Quant(nn.Module):
    def __init__(self, num_classes=1000, weight_bit_width=1, act_bit_width=1):
        super(ResNet50Quant, self).__init__()

        self.in_planes = 64
        weight_quant = CommonWeightQuant
        act_quant = CommonActQuant

        self.conv1 = QuantConv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False, weight_quant=weight_quant,
                                 bit_width=weight_bit_width)
        self.bn1 = BatchNorm2d(64)
        self.relu = QuantReLU(act_quant=act_quant, bit_width=act_bit_width),
        self.maxpool = QuantMaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BottleneckQuant, 64, 3, stride=1, weight_bit_width=weight_bit_width,
                                       act_bit_width=act_bit_width)
        self.layer2 = self._make_layer(BottleneckQuant, 128, 4, stride=2, weight_bit_width=weight_bit_width,
                                       act_bit_width=act_bit_width)
        self.layer3 = self._make_layer(BottleneckQuant, 256, 6, stride=2, weight_bit_width=weight_bit_width,
                                       act_bit_width=act_bit_width)
        self.layer4 = self._make_layer(BottleneckQuant, 512, 3, stride=2, weight_bit_width=weight_bit_width,
                                       act_bit_width=act_bit_width)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = QuantLinear(512 * 4, num_classes, weight_quant=weight_quant, bit_width=weight_bit_width, bias=False)

    def _make_layer(self, block, planes, num_blocks, stride, weight_bit_width, act_bit_width):
        layers = []
        layers.append(block(self.in_planes, planes, stride, weight_bit_width, act_bit_width))
        self.in_planes = planes * 4
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, 1, weight_bit_width, act_bit_width))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out