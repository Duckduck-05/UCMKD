from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn

_N_CLASSES = 8  # RAVVDESS has 8 emotion classes


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, modality, num_classes=1000, num_frame=10, pool='avgpool',
                 zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, norm_layer=None):
        super(ResNet, self).__init__()
        self.modality = modality
        self.pool = pool
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        if modality == 'audio':
            self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        elif modality == 'visual':
            self.conv1 = nn.Conv2d(3 * num_frame, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            raise NotImplementedError('Incorrect modality, should be audio or visual but got {}'.format(modality))
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        if self.pool == 'avgpool':
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
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
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward_encoder(self, x):
        x = x.float()

        if self.modality == 'visual':
            if x.dim() == 5:
                (B, C, T, H, W) = x.size()
                x = x.permute(0, 2, 1, 3, 4).contiguous()
                x = x.view(B, C * T, H, W)
        else:  # audio
            if x.dim() == 5:
                x = x.squeeze(2)
            elif x.dim() == 3:
                x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        f0 = x
        x = self.maxpool(x)

        x = self.layer1(x)
        f1 = x
        x = self.layer2(x)
        f2 = x
        x = self.layer3(x)
        f3 = x
        x = self.layer4(x)
        f4 = x

        x_512 = self.avgpool(x)
        feature_vector = x_512.reshape(x_512.shape[0], -1)

        return feature_vector, [f0, f1, f2, f3, f4]

    def forward_head(self, feature_vector):
        return self.fc(feature_vector)

    def forward(self, x):
        feature, feature_maps = self.forward_encoder(x)
        logits = self.forward_head(feature)
        return logits, feature, feature_maps


def _resnet(arch, block, layers, modality, num_classes, num_frame):
    model = ResNet(block, layers, modality, num_classes=num_classes, num_frame=num_frame)
    return model


class ViTImageNet(nn.Module):
    """ViT-B/16 backbone for image modality.

    Interface:
      forward(x)         → (logits, cls_token_768, [f0, f1, f2, f3, f4])
      forward_encoder(x) → (cls_token_768, [f0, f1, f2, f3, f4])
      forward_head(feat) → logits
      fc(feat)           → logits
      feature_dim        = 768
    """

    feature_dim: int = 768

    def __init__(self, num_classes: int = _N_CLASSES, num_frame: int = 1):
        super().__init__()
        from torchvision.models import vit_b_16, ViT_B_16_Weights

        self.num_frame = num_frame
        vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        vit.heads.head = nn.Linear(768, num_classes)
        self._vit = vit

    def fc(self, x: torch.Tensor) -> torch.Tensor:
        return self._vit.heads.head(x)

    def forward_encoder(self, x: torch.Tensor):
        if x.dim() == 5:
            x = x.mean(dim=2)          # (B, 3, H, W)
        x = x.float()
        vit = self._vit

        x_emb = vit._process_input(x)
        n = x_emb.shape[0]
        cls = vit.class_token.expand(n, -1, -1)
        h = torch.cat([cls, x_emb], dim=1)
        h = vit.encoder.dropout(h)

        feats = []
        for block in vit.encoder.layers:
            h = block(h)
            feats.append(h)

        h = vit.encoder.ln(h)
        cls_token = h[:, 0]            # (B, 768)

        def _to_2d(feat):
            patches = feat[:, 1:]
            B_ = patches.shape[0]
            return patches.transpose(1, 2).reshape(B_, 768, 14, 14)

        f0 = _to_2d(feats[2])
        f1 = _to_2d(feats[5])
        f2 = _to_2d(feats[8])
        f3 = _to_2d(feats[11])
        f4 = _to_2d(h)

        return cls_token, [f0, f1, f2, f3, f4]

    def forward_head(self, feature_vector: torch.Tensor) -> torch.Tensor:
        return self._vit.heads.head(feature_vector)

    def forward(self, x: torch.Tensor):
        cls_token, feature_maps = self.forward_encoder(x)
        logits = self.forward_head(cls_token)
        return logits, cls_token, feature_maps


class ViTAudioNet(nn.Module):
    """ViT-Small/16 (timm) backbone for audio modality (1-channel mel spectrogram).

    Uses timm's vit_small_patch16_224 (384-dim) with ImageNet pretrained weights
    and projects features to 768-dim to match ViTImageNet's interface.

    Interface:
      forward(x)         → (logits, cls_token, [f0, f1, f2, f3, f4])
      forward_encoder(x) → (cls_token, [f0, f1, f2, f3, f4])
      forward_head(feat) → logits

    - cls_token, fi : (B, 768) after 384→768 projection
    - spatial maps  : (B, 768, 14, 14)

    Input x: (B, F, T) mel spectrogram — unsqueezed to (B, 1, F, T),
    then F.interpolate to (B, 1, 224, 224).
    """
    _DIM = 384      # vit_small hidden dim
    _OUT_DIM = 768  # projected dim (matches ViTImageNet)

    def __init__(self, num_classes: int = 28, num_frame: int = 1):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            'vit_small_patch16_224',
            pretrained=True,   # ImageNet weights → fine-tune (AST approach)
            in_chans=1,        # timm averages 3-ch patch embed → 1-ch automatically
            num_classes=0,     # return CLS token (384-dim), not logits
            drop_path_rate=0.1,  # stochastic depth regularization
        )
        # Freeze patch embedding, positional embedding, and lower 8 of 12 blocks.
        # Fine-tune only the top 4 blocks + norm (preserves pre-trained features,
        # prevents overfitting on small RAVDESS dataset ~1k train samples).
        for p in self.backbone.patch_embed.parameters():
            p.requires_grad = False
        self.backbone.pos_embed.requires_grad = False
        self.backbone.cls_token.requires_grad = False
        for i, block in enumerate(self.backbone.blocks):
            if i < 8:
                for p in block.parameters():
                    p.requires_grad = False
        self.proj = nn.Linear(self._DIM, self._OUT_DIM)
        self.fc_head = nn.Linear(self._OUT_DIM, num_classes)

    def fc(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_head(x)

    def forward_encoder(self, x: torch.Tensor):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = x.float()
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        vit = self.backbone
        feats = []
        handles = [blk.register_forward_hook(lambda m, i, o: feats.append(o))
                   for blk in vit.blocks]
        cls_384 = vit(x)   # (B, 384) — CLS token after norm, via timm forward
        for h in handles:
            h.remove()

        cls_token = self.proj(cls_384)  # (B, 768)

        def _to_2d(feat):
            patches = feat[:, 1:]  # (B, 196, 384)
            proj = self.proj(patches)  # (B, 196, 768)
            return proj.transpose(1, 2).reshape(proj.shape[0], self._OUT_DIM, 14, 14)

        f0 = _to_2d(feats[2])
        f1 = _to_2d(feats[5])
        f2 = _to_2d(feats[8])
        f3 = _to_2d(feats[10])
        f4 = _to_2d(feats[11])

        return cls_token, [f0, f1, f2, f3, f4]

    def forward_head(self, feature_vector: torch.Tensor) -> torch.Tensor:
        return self.fc_head(feature_vector)

    def forward(self, x: torch.Tensor):
        cls_token, feature_maps = self.forward_encoder(x)
        logits = self.forward_head(cls_token)
        return logits, cls_token, feature_maps


class ViTLImageNet(nn.Module):
    """ViT-L/16 backbone for image modality.

    Interface:
      forward(x)         → (logits, cls_token_1024, [f0, f1, f2, f3, f4])
      forward_encoder(x) → (cls_token_1024, [f0, f1, f2, f3, f4])
      forward_head(feat) → logits
      fc(feat)           → logits
      feature_dim        = 1024

    - f0..f3: patch-token maps from blocks 5, 11, 17, 23  → (B, 1024, 14, 14)
    - f4    : patch-token map after the final LayerNorm    → (B, 1024, 14, 14)
    """

    feature_dim: int = 1024

    def __init__(self, num_classes: int = _N_CLASSES, num_frame: int = 1):
        super().__init__()
        from torchvision.models import vit_l_16, ViT_L_16_Weights

        self.num_frame = num_frame
        vit = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1)
        vit.heads.head = nn.Linear(1024, num_classes)
        self._vit = vit

    def fc(self, x: torch.Tensor) -> torch.Tensor:
        return self._vit.heads.head(x)

    def forward_encoder(self, x: torch.Tensor):
        if x.dim() == 5:
            x = x.mean(dim=2)          # (B, 3, H, W)
        x = x.float()
        vit = self._vit

        x_emb = vit._process_input(x)
        n = x_emb.shape[0]
        cls = vit.class_token.expand(n, -1, -1)
        h = torch.cat([cls, x_emb], dim=1)
        h = vit.encoder.dropout(h)

        feats = []
        for block in vit.encoder.layers:
            h = block(h)
            feats.append(h)

        h = vit.encoder.ln(h)
        cls_token = h[:, 0]            # (B, 1024)

        def _to_2d(feat):
            patches = feat[:, 1:]
            B_ = patches.shape[0]
            return patches.transpose(1, 2).reshape(B_, 1024, 14, 14)

        f0 = _to_2d(feats[5])
        f1 = _to_2d(feats[11])
        f2 = _to_2d(feats[17])
        f3 = _to_2d(feats[23])
        f4 = _to_2d(h)

        return cls_token, [f0, f1, f2, f3, f4]

    def forward_head(self, feature_vector: torch.Tensor) -> torch.Tensor:
        return self._vit.heads.head(feature_vector)

    def forward(self, x: torch.Tensor):
        cls_token, feature_maps = self.forward_encoder(x)
        logits = self.forward_head(cls_token)
        return logits, cls_token, feature_maps


class ViTLAudioNet(nn.Module):
    """ViT-L/16 (timm) backbone for audio modality (1-channel mel spectrogram).

    Uses timm's vit_large_patch16_224 (1024-dim) with ImageNet pretrained weights.
    Freezes patch embedding, pos embedding, cls token, and the lower 16/24 blocks
    to prevent overfitting on the small RAVDESS dataset.

    Interface:
      forward(x)         → (logits, cls_token, [f0, f1, f2, f3, f4])
      forward_encoder(x) → (cls_token, [f0, f1, f2, f3, f4])
      forward_head(feat) → logits

    - cls_token, fi : (B, 1024)
    - spatial maps  : (B, 1024, 14, 14)

    Input x: (B, F, T) mel spectrogram — unsqueezed to (B, 1, F, T),
    then F.interpolate to (B, 1, 224, 224).
    """
    _DIM = 1024

    def __init__(self, num_classes: int = _N_CLASSES, num_frame: int = 1):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            'vit_large_patch16_224',
            pretrained=True,
            in_chans=1,
            num_classes=0,     # return CLS token (1024-dim), not logits
            drop_path_rate=0.1,
        )
        # Freeze patch embedding, positional embedding, cls token,
        # and the lower 16 of 24 blocks (similar proportion to ViTAudioNet's 8/12).
        for p in self.backbone.patch_embed.parameters():
            p.requires_grad = False
        self.backbone.pos_embed.requires_grad = False
        self.backbone.cls_token.requires_grad = False
        for i, block in enumerate(self.backbone.blocks):
            if i < 16:
                for p in block.parameters():
                    p.requires_grad = False
        self.fc_head = nn.Linear(self._DIM, num_classes)

    def fc(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_head(x)

    def forward_encoder(self, x: torch.Tensor):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = x.float()
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        vit = self.backbone
        feats = []
        handles = [blk.register_forward_hook(lambda m, i, o: feats.append(o))
                   for blk in vit.blocks]
        cls_1024 = vit(x)   # (B, 1024) — CLS token after norm, via timm forward
        for h in handles:
            h.remove()

        def _to_2d(feat):
            patches = feat[:, 1:]  # (B, 196, 1024)
            return patches.transpose(1, 2).reshape(patches.shape[0], self._DIM, 14, 14)

        f0 = _to_2d(feats[5])
        f1 = _to_2d(feats[11])
        f2 = _to_2d(feats[17])
        f3 = _to_2d(feats[21])
        f4 = _to_2d(feats[23])

        return cls_1024, [f0, f1, f2, f3, f4]

    def forward_head(self, feature_vector: torch.Tensor) -> torch.Tensor:
        return self.fc_head(feature_vector)

    def forward(self, x: torch.Tensor):
        cls_token, feature_maps = self.forward_encoder(x)
        logits = self.forward_head(cls_token)
        return logits, cls_token, feature_maps


class AudioNet(nn.Module):
    """AudioNet — supports resnet18, resnet50, vit_s_16, and vit_l_16."""

    def __init__(self, args):
        super(AudioNet, self).__init__()
        self.arch = args.audio_arch
        if self.arch == 'vit_l_16':
            self.backbone = ViTLAudioNet(num_classes=_N_CLASSES, num_frame=args.num_frame)
            self._feature_dim = ViTLAudioNet._DIM  # 1024
        elif self.arch == 'vit_s_16':
            self.backbone = ViTAudioNet(num_classes=_N_CLASSES, num_frame=args.num_frame)
            self._feature_dim = ViTAudioNet._OUT_DIM  # 768
        else:
            if self.arch == 'resnet18':
                layers = [2, 2, 2, 2]
            elif self.arch == 'resnet50':
                layers = [3, 4, 6, 3]
            else:
                raise ValueError(f'Unknown audio_arch: {self.arch}')
            self.backbone = _resnet('resnet_x', BasicBlock, layers, modality='audio', num_classes=_N_CLASSES, num_frame=args.num_frame)
            self._feature_dim = 512

    @property
    def feature_dim(self):
        return self._feature_dim

    def fc(self, x):
        return self.backbone.fc(x)

    def forward_encoder(self, x):
        return self.backbone.forward_encoder(x)

    def forward_head(self, feature_vector):
        return self.backbone.forward_head(feature_vector)

    def forward(self, x):
        return self.backbone(x)


class ImageNet(nn.Module):
    """Wrapper: supports resnet18, resnet50, vit_b_16, and vit_l_16."""

    def __init__(self, args):
        super(ImageNet, self).__init__()
        self.arch = args.image_arch
        if self.arch == 'vit_l_16':
            self.backbone = ViTLImageNet(num_classes=_N_CLASSES, num_frame=args.num_frame)
            self._feature_dim = ViTLImageNet.feature_dim  # 1024
        elif self.arch == 'vit_b_16':
            self.backbone = ViTImageNet(num_classes=_N_CLASSES, num_frame=args.num_frame)
            self._feature_dim = 768
        else:
            if self.arch == 'resnet18':
                layers = [2, 2, 2, 2]
            elif self.arch == 'resnet50':
                layers = [3, 4, 6, 3]
            else:
                raise ValueError(f'Unknown image_arch: {self.arch}')
            self.backbone = _resnet('resnet_x', BasicBlock, layers, modality='visual', num_classes=_N_CLASSES, num_frame=args.num_frame)
            self._feature_dim = 512

    @property
    def feature_dim(self):
        return self._feature_dim

    def fc(self, x):
        return self.backbone.fc(x)

    def forward_encoder(self, x):
        return self.backbone.forward_encoder(x)

    def forward_head(self, feature_vector):
        return self.backbone.forward_head(feature_vector)

    def forward(self, x):
        return self.backbone(x)


class FCReg(nn.Module):
    """Convolutional regression"""

    def __init__(self, s_C1, s_C2, use_relu=True):
        super(FCReg, self).__init__()
        self.use_relu = use_relu
        self.fc = nn.Linear(s_C1, s_C2)
        self.bn = nn.BatchNorm1d(s_C2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc(x)
        if self.use_relu:
            return self.relu(self.bn(x))
        else:
            return self.bn(x)
