"""Proxy heads for C²KD (ResNet: Tea/Stu, ViT: TeaViT/StuViT)."""

import torch.nn as nn


def _conv_1x1_bn(ch):
    return nn.Sequential(
        nn.Conv2d(ch, ch, 1, bias=False),
        nn.BatchNorm2d(ch),
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
    """Teacher proxy — f4 is layer4 output (512ch)."""
    def __init__(self):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.layer = _conv_1x1_bn(512)
        self.fc = nn.Linear(512, 8)
        _init(self)

    def forward(self, tea):
        x = self.avgpool(tea[4])
        x = self.layer(x).view(x.size(0), -1)
        return self.fc(x)


class Stu(nn.Module):
    """Student proxy — f4 is layer4 output (512ch)."""
    def __init__(self):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.layer = _conv_1x1_bn(512)
        self.fc = nn.Linear(512, 8)
        _init(self)

    def forward(self, stu):
        x = self.avgpool(stu[4])
        x = self.layer(x).view(x.size(0), -1)
        return self.fc(x)


class TeaViT(nn.Module):
    """Teacher proxy for ViT backbones."""
    def __init__(self, feat_dim: int = 768, num_classes: int = 8):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.layer = _conv_1x1_bn(feat_dim)
        self.fc = nn.Linear(feat_dim, num_classes)
        _init(self)

    def forward(self, tea):
        x = self.avgpool(tea[4])
        x = self.layer(x).view(x.size(0), -1)
        return self.fc(x)


class StuViT(nn.Module):
    """Student proxy for ViT backbones."""
    def __init__(self, feat_dim: int = 768, num_classes: int = 8):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.layer = _conv_1x1_bn(feat_dim)
        self.fc = nn.Linear(feat_dim, num_classes)
        _init(self)

    def forward(self, stu):
        x = self.avgpool(stu[4])
        x = self.layer(x).view(x.size(0), -1)
        return self.fc(x)
