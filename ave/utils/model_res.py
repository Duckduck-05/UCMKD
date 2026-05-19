from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn


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
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
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


class ResNet(nn.Module):

    def __init__(self, block, layers, modality, num_classes=1000, num_frame=10, pool='avgpool', zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        self.modality = modality
        self.pool = pool
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        if modality == 'audio':
            self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        elif modality == 'visual':
            self.conv1 = nn.Conv2d(3 * num_frame, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
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

            self.fc = nn.Linear(512 * block.expansion, num_classes)  # 8192

        # if modality == 'audio':
        #     self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #     self.fc = nn.Linear(512 * block.expansion, num_classes)  # 8192
        # elif modality == 'visual':
        #     self.avgpool = nn.AdaptiveAvgPool3d(1)
        #     self.fc = nn.Linear(512 * block.expansion, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
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
        if self.modality == 'visual':
            (B, C, T, H, W) = x.size()
            x = x.permute(0, 2, 1, 3, 4).contiguous()
            x = x.view(B, C * T, H, W)
        else:
            x = x.unsqueeze(1)
        x = x.float()

        # --- Backbone ---
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
        
        # --- Pooling & Flatten ---
        x_512 = self.avgpool(x)
        feature_vector = x_512.reshape(x_512.shape[0], -1) #  phi
        
        return feature_vector, [f0, f1, f2, f3, f4]

    def forward_head(self, feature_vector):
        logits = self.fc(feature_vector)
        return logits
    
    def forward(self, x):
        feature, feature_maps = self.forward_encoder(x)

        logits = self.forward_head(feature)

        return logits, feature, feature_maps

    
    # def forward(self, x):
    #     if self.modality == 'visual':
    #         (B, C, T, H, W) = x.size()
    #         x = x.permute(0, 2, 1, 3, 4).contiguous()
    #         x = x.view(B, C * T, H, W)
    #     else:
    #         x = x.unsqueeze(1)
    #     # x = x.unsqueeze(1)
    #     x = x.float()
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.relu(x)
    #     f0 = x
    #     x = self.maxpool(x)

    #     x = self.layer1(x)
    #     f1 = x
    #     x = self.layer2(x)
    #     f2 = x
    #     x = self.layer3(x)
    #     f3 = x
    #     x_512 = self.avgpool(self.layer4(x))
    #     x_512 = x_512.reshape(x_512.shape[0], -1)
    #     f4 = x
    #     out = self.fc(x_512)

    #     return out, x_512, [f0, f1, f2, f3, f4]


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
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



def _resnet(arch, block, layers, modality, num_classes, num_frame):
    model = ResNet(block, layers, modality, num_classes=num_classes, num_frame=num_frame)
    return model


class ViTImageNet(nn.Module):
    """ViT-B/16 backbone for image modality.

    Exposes the same interface as the ResNet backbone:
      forward(x)         → (logits, cls_token, [f0, f1, f2, f3, f4])
      forward_encoder(x) → (cls_token, [f0, f1, f2, f3, f4])
      forward_head(feat) → logits

    - logits    : (B, num_classes)
    - cls_token : (B, 768)
    - f0..f3    : (B, 768, 14, 14) patch-token maps from blocks 2, 5, 8, 11
    - f4        : (B, 768, 14, 14) patch-token map after the final LayerNorm
    """

    def __init__(self, num_classes: int = 28, num_frame: int = 1):
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


class ViTLImageNet(nn.Module):
    """ViT-L/16 backbone for image modality.

    Same interface as ViTImageNet but with ViT-L/16:
    - cls_token : (B, 1024)
    - f0..f3    : (B, 1024, 14, 14) patch-token maps from blocks 5, 11, 17, 23
    - f4        : (B, 1024, 14, 14) patch-token map after the final LayerNorm
    """

    def __init__(self, num_classes: int = 28, num_frame: int = 1):
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

class ViTAudioNet(nn.Module):
    """
    Audio Spectrogram Transformer (AST) cho audio input shape [B, 257, 1024].
    Uses vit_small_patch16 as backbone, trained from scratch.
    Output interface matches ResNet: forward() → (logits, feature_512, feature_maps)
    """

    def __init__(self, args, num_classes: int = 28):
        super().__init__()
        import timm

        # Patch spectrogram 2D, input: [B, 1, H, W]
        # timm's vit_small_patch16 expects 224x224 by default
        # Override img_size to fit spectrogram [257, 1024]
        # Crop to [256, 1024] to be divisible by patch_size=16
        self.target_H = 256  # crop 257 → 256 (divisible by 16)
        self.target_W = 1024

        self.backbone = timm.create_model(
            'vit_small_patch16_224',
            pretrained=True,   # ImageNet pretrained → fine-tune on audio (AST approach)
            in_chans=1,        # timm averages 3-ch patch embed weights → 1-ch
            img_size=(self.target_H, self.target_W),  # timm interpolates pos_embed automatically
            num_classes=0,     # remove classification head, return raw features
        )
        # vit_small embed_dim = 384
        vit_dim = self.backbone.embed_dim  # 384

        # Project 384 → 768 to match ViTImageNet cls_token dim
        self.proj = nn.Sequential(
            nn.Linear(vit_dim, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(inplace=True),
        )

        # Classifier head
        self.fc = nn.Linear(768, num_classes)

    def forward_encoder(self, x):
        """
        x: [B, 257, 1024] (raw spectrogram, no channel dim)
        returns: feature (B, 768), feature_maps placeholder
        """
        # Add channel dim → [B, 1, 257, 1024]
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = x.float()

        # Resize to (target_H, target_W) regardless of input size
        x = F.interpolate(x, size=(self.target_H, self.target_W), mode='bilinear', align_corners=False)

        # Forward through ViT backbone, extract CLS token
        # timm returns (B, num_patches+1, dim) via forward_features
        tokens = self.backbone.forward_features(x)  # (B, num_patches+1, 384)
        cls_token = tokens[:, 0, :]                 # (B, 384)

        # Project → 512
        feature = self.proj(cls_token)              # (B, 512)

        # Use patch tokens as feature maps (analogous to f0..f4 in ResNet)
        patch_tokens = tokens[:, 1:, :]             # (B, num_patches, 384)
        # num_patches = (256/16) * (1024/16) = 16 * 64 = 1024
        # Reshape → (B, 384, 16, 64) to match spatial feature map format
        B, N, D = patch_tokens.shape
        fmap = patch_tokens.transpose(1, 2).reshape(B, D, 16, 64)
        feature_maps = [fmap, fmap, fmap, fmap, fmap]  # placeholder 5 levels

        return feature, feature_maps

    def forward_head(self, feature):
        return self.fc(feature)

    def forward(self, x):
        feature, feature_maps = self.forward_encoder(x)
        logits = self.forward_head(feature)
        return logits, feature, feature_maps


class ViTLAudioNet(nn.Module):
    """ViT-L/16 backbone for audio modality (AST-style).

    Input: [B, 257, 1024] spectrogram.
    Output interface same as ViTAudioNet:
      forward() → (logits, feature, feature_maps)
    - feature     : (B, 1024)
    - feature_maps: list of 5 x (B, 1024, 16, 64)
    """

    def __init__(self, args, num_classes: int = 28):
        super().__init__()
        import timm

        self.target_H = 256
        self.target_W = 1024

        self.backbone = timm.create_model(
            'vit_large_patch16_224',
            pretrained=True,
            in_chans=1,
            img_size=(self.target_H, self.target_W),
            num_classes=0,
        )
        vit_dim = self.backbone.embed_dim  # 1024

        self.fc = nn.Linear(vit_dim, num_classes)

    def forward_encoder(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = x.float()
        x = F.interpolate(x, size=(self.target_H, self.target_W), mode='bilinear', align_corners=False)

        tokens = self.backbone.forward_features(x)  # (B, num_patches+1, 1024)
        cls_token = tokens[:, 0, :]                 # (B, 1024)

        patch_tokens = tokens[:, 1:, :]             # (B, 1024, 1024)
        B, N, D = patch_tokens.shape
        fmap = patch_tokens.transpose(1, 2).reshape(B, D, 16, 64)
        feature_maps = [fmap, fmap, fmap, fmap, fmap]

        return cls_token, feature_maps

    def forward_head(self, feature):
        return self.fc(feature)

    def forward(self, x):
        feature, feature_maps = self.forward_encoder(x)
        logits = self.forward_head(feature)
        return logits, feature, feature_maps


class AudioNet(nn.Module):
    def __init__(self, args):
        super(AudioNet, self).__init__()
        self.arch = args.audio_arch

        if self.arch == 'vit_l_16':
            self.backbone = ViTLAudioNet(args, num_classes=28)
            self._feature_dim = 1024
        elif self.arch == 'vit_s_16':
            self.backbone = ViTAudioNet(args, num_classes=28)
            self._feature_dim = 768
        else:
            if self.arch == 'resnet18':
                layers = [2, 2, 2, 2]
            elif self.arch == 'resnet50':
                layers = [3, 4, 6, 3]
            self.backbone = _resnet('resnet_x', BasicBlock, layers,
                                    modality='audio', num_classes=28,
                                    num_frame=args.num_frame)
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
        return x

class ImageNet(nn.Module):
    """ImageNet — supports resnet18, resnet50, and vit_b_16."""

    def __init__(self, args):
        super(ImageNet, self).__init__()
        self.arch = args.image_arch
        if self.arch == 'vit_l_16':
            self.backbone = ViTLImageNet(num_classes=28, num_frame=args.num_frame)
            self._feature_dim = 1024
        elif self.arch == 'vit_b_16':
            self.backbone = ViTImageNet(num_classes=28, num_frame=args.num_frame)
            self._feature_dim = 768
        else:
            if self.arch == 'resnet18':
                layers = [2, 2, 2, 2]
            elif self.arch == 'resnet50':
                layers = [3, 4, 6, 3]
            else:
                raise ValueError(f'Unknown image_arch: {self.arch}')
            self.backbone = _resnet('resnet_x', BasicBlock, layers, modality='visual', num_classes=28, num_frame=args.num_frame)
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

