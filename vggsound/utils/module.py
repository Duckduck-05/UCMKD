"""Proxy heads for C²KD (Tea/Stu for ResNet backbones)."""

import torch
import torch.nn as nn


def _conv_1x1_bn(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.1, inplace=True),
    )


def _init(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class Tea(nn.Module):
    """Teacher proxy for C²KD.

    tea_type: 0 = image teacher (3D temporal pooling), 1 = audio teacher (2D pooling).
    """
    def __init__(self, tea_type: int):
        super().__init__()
        self.tea_type = tea_type
        self.avgpool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool3d = nn.AdaptiveAvgPool3d(1)
        self.layer = _conv_1x1_bn(256, 256)
        self.fc = nn.Linear(256, 50)
        _init(self)

    def forward(self, tea):
        x = tea[-1]
        if self.tea_type == 0:
            B, C, H, W = x.shape
            x = x.view(B // 3, 3, C, H, W).permute(0, 2, 1, 3, 4)
            x = self.avgpool3d(x)
        else:
            x = self.avgpool2d(x)
        x = self.layer(x.view(x.size(0), -1, 1, 1))
        x = x.view(x.size(0), -1)
        return self.fc(x)


class Stu(nn.Module):
    """Student proxy for C²KD.

    tea_type: 0 = image student (3D temporal pooling), 1 = audio student (2D pooling).
    """
    def __init__(self, tea_type: int):
        super().__init__()
        self.tea_type = tea_type
        self.avgpool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool3d = nn.AdaptiveAvgPool3d(1)
        self.layer = _conv_1x1_bn(256, 256)
        self.fc = nn.Linear(256, 50)
        _init(self)

    def forward(self, stu):
        x = stu[-1]
        if self.tea_type == 0:
            B, C, H, W = x.shape
            x = x.view(B // 3, 3, C, H, W).permute(0, 2, 1, 3, 4)
            x = self.avgpool3d(x)
        else:
            x = self.avgpool2d(x)
        x = self.layer(x.view(x.size(0), -1, 1, 1))
        x = x.view(x.size(0), -1)
        return self.fc(x)
