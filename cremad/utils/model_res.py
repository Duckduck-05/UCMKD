import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.backbone import ResNet

_N_CLASSES = 6  # CREMAD has 6 emotion classes


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
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


def _resnet(arch, block, layers, modality, num_classes, **kwargs):
    model = ResNet(block, layers, modality, num_classes=num_classes, **kwargs)
    return model


# ---------------------------------------------------------------------------
# ViT-based backbones
# ---------------------------------------------------------------------------

class ViTImageNet(nn.Module):
    """ViT-B/16 backbone for CREMAD video modality.

    Input: (B, T, C, H, W) — T frames per video clip.
    Temporal pooling: mean over T frames → (B, 3, H, W) before feeding ViT.

    Interface:
      forward(x)         → (logits, cls_token_768, [f0, f1, f2, f3, f4])
      forward_encoder(x) → (cls_token_768, [f0, f1, f2, f3, f4])
      forward_head(feat) → logits
      fc(feat)           → logits
      feature_dim        = 768
    """

    feature_dim: int = 768

    def __init__(self, num_classes: int = _N_CLASSES):
        super().__init__()
        from torchvision.models import vit_b_16, ViT_B_16_Weights

        vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        vit.heads.head = nn.Linear(768, num_classes)
        self._vit = vit

    def fc(self, x: torch.Tensor) -> torch.Tensor:
        return self._vit.heads.head(x)

    def _temporal_pool(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C, H, W) or (B, C, H, W) → (B, 3, H, W)."""
        if x.dim() == 5:
            x = x.mean(dim=1)   # average over T frames (dim 1 for CREMAD)
        return x.float()

    def forward_encoder(self, x: torch.Tensor):
        x = self._temporal_pool(x)
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
        cls_token = h[:, 0]   # (B, 768)

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
    """Audio Spectrogram Transformer for CREMAD audio input shape [B, 257, 1024].

    Uses vit_small_patch16 backbone. Projects 384 → 768 to match ViTImageNet.

    Interface:
      forward(x)         → (logits, feature_768, feature_maps)
      forward_encoder(x) → (feature_768, feature_maps)
      forward_head(feat) → logits
      fc(feat)           → logits
      feature_dim        = 768
    """

    feature_dim: int = 768
    _H: int = 256   # crop 257 → 256 for patch_size=16 divisibility
    _W: int = 1024

    def __init__(self, num_classes: int = _N_CLASSES):
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            'vit_small_patch16_224',
            pretrained=True,   # ImageNet pretrained → fine-tune on audio (AST approach)
            in_chans=1,        # timm averages 3-ch patch embed weights → 1-ch
            img_size=(self._H, self._W),  # timm interpolates pos_embed automatically
            num_classes=0,
        )
        vit_dim = self.backbone.embed_dim  # 384

        self.proj = nn.Sequential(
            nn.Linear(vit_dim, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(inplace=True),
        )
        self.fc_head = nn.Linear(768, num_classes)

    def fc(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_head(x)

    def forward_encoder(self, x: torch.Tensor):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = x.float()
        # Resize to (H, W) regardless of input size — handles any spectrogram shape
        x = F.interpolate(x, size=(self._H, self._W), mode='bilinear', align_corners=False)

        tokens = self.backbone.forward_features(x)   # (B, N+1, 384)
        cls_token = tokens[:, 0, :]                  # (B, 384)
        feature = self.proj(cls_token)               # (B, 768)

        patch_tokens = tokens[:, 1:, :]
        B, N, D = patch_tokens.shape
        fmap = patch_tokens.transpose(1, 2).reshape(B, D, 16, 64)
        feature_maps = [fmap, fmap, fmap, fmap, fmap]

        return feature, feature_maps

    def forward_head(self, feature: torch.Tensor) -> torch.Tensor:
        return self.fc_head(feature)

    def forward(self, x: torch.Tensor):
        feature, feature_maps = self.forward_encoder(x)
        logits = self.forward_head(feature)
        return logits, feature, feature_maps


# ---------------------------------------------------------------------------
# ResNet-based backbones (original CREMAD logic, preserved)
# ---------------------------------------------------------------------------

class _ResNetImageNet(nn.Module):
    """ResNet visual backbone with 3D temporal pooling for CREMAD video."""

    feature_dim: int = 512

    def __init__(self, layers, num_classes=_N_CLASSES):
        super().__init__()
        self.backbone = ResNet(BasicBlock, layers, modality='visual',
                               num_classes=num_classes)
        self.head_video = nn.Linear(512, num_classes)

    def fc(self, feature_vector):
        return self.head_video(feature_vector)

    def forward_encoder(self, x):
        B = x.size(0)
        T = x.size(1)
        x_flat = x.view(B * T, x.size(2), x.size(3), x.size(4))
        bb = self.backbone
        z = bb.maxpool(bb.relu(bb.bn1(bb.conv1(x_flat))))
        f1 = bb.layer1(z)
        f2 = bb.layer2(f1)
        f3 = bb.layer3(f2)
        f4 = bb.layer4(f3)

        # Average over T frames: (B*T, C, H, W) → (B, C, H, W)
        def _tavg(ft):
            _, C, H, W = ft.shape
            return ft.view(B, T, C, H, W).mean(1)

        feature_maps = [_tavg(f1), _tavg(f2), _tavg(f3), _tavg(f4)]

        # 3D temporal pooling for feature vector (preserves original behaviour)
        v = f4.view(B, T, f4.size(1), f4.size(2), f4.size(3))
        v = v.permute(0, 2, 1, 3, 4)                # (B, C, T, H', W')
        v = F.adaptive_avg_pool3d(v, 1)
        feature_vector = torch.flatten(v, 1)         # (B, 512)
        return feature_vector, feature_maps

    def forward_head(self, feature_vector):
        return self.head_video(feature_vector)

    def forward(self, x):
        feature_vector, feature_maps = self.forward_encoder(x)
        logits = self.forward_head(feature_vector)
        return logits, feature_vector, feature_maps


class _ResNetAudioNet(nn.Module):
    """ResNet audio backbone for CREMAD spectrogram."""

    feature_dim: int = 512

    def __init__(self, layers, num_classes=_N_CLASSES):
        super().__init__()
        self.backbone = ResNet(BasicBlock, layers, modality='audio',
                               num_classes=num_classes)
        self.head_audio = nn.Linear(512, num_classes)

    def fc(self, feature_vector):
        return self.head_audio(feature_vector)

    def forward_encoder(self, x):
        bb = self.backbone
        z = bb.maxpool(bb.relu(bb.bn1(bb.conv1(x))))
        f1 = bb.layer1(z)
        f2 = bb.layer2(f1)
        f3 = bb.layer3(f2)
        f4 = bb.layer4(f3)
        pooled = F.adaptive_avg_pool2d(f4, 1)
        feature_vector = torch.flatten(pooled, 1)    # (B, 512)
        return feature_vector, [f1, f2, f3, f4]


class ViTAudioNet(nn.Module):
    """ViT-B/16 backbone for audio modality (1-channel mel spectrogram).

    Exposes the same interface as the ResNet backbone:
      forward(x)         → (logits, cls_token, [f0, f1, f2, f3, f4])
      forward_encoder(x) → (cls_token, [f0, f1, f2, f3, f4])
      forward_head(feat) → logits

    - logits    : (B, num_classes)
    - cls_token : (B, 768)
    - f0..f3    : (B, 768, 14, 14) patch-token maps from blocks 2, 5, 8, 11
    - f4        : (B, 768, 14, 14) patch-token map after the final LayerNorm

    Input x: (B, F, T) mel spectrogram — unsqueezed to (B, 1, F, T) internally,
    then resized to (B, 1, 224, 224) for ViT.
    """

    def __init__(self, num_classes: int = 6, num_frame: int = 1):
        super().__init__()
        from torchvision.models import vit_b_16, ViT_B_16_Weights

        vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        # Adapt patch projection to accept 1-channel (spectrogram) input
        old_proj = vit.conv_proj  # Conv2d(3, 768, 16, 16)
        new_proj = nn.Conv2d(1, 768, kernel_size=16, stride=16)
        with torch.no_grad():
            new_proj.weight.copy_(old_proj.weight.mean(dim=1, keepdim=True))
            new_proj.bias.copy_(old_proj.bias)
        vit.conv_proj = new_proj
        vit.heads.head = nn.Linear(768, num_classes)
        self._vit = vit

    def fc(self, x: torch.Tensor) -> torch.Tensor:
        return self._vit.heads.head(x)

    def forward_encoder(self, x: torch.Tensor):
        # x: (B, F, T) → unsqueeze → (B, 1, F, T), then resize to 224×224
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = x.float()
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

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
        cls_token = h[:, 0]  # (B, 768)

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


class ViTSAudioNet(nn.Module):
    """ViT-S/16 backbone for audio modality (1-channel mel spectrogram) using timm.

    Same interface as ViTAudioNet:
      forward(x)         → (logits, cls_token, feature_maps)
      forward_encoder(x) → (cls_token, feature_maps)
      forward_head(feat) → logits

    - logits    : (B, num_classes)
    - cls_token : (B, 384)

    Input x: (B, F, T) mel spectrogram — resized to (B, 1, 224, 224) internally.
    """

    def __init__(self, num_classes: int = 6):
        super().__init__()
        import timm
        vit = timm.create_model('vit_small_patch16_224', pretrained=True)
        # Adapt patch projection to accept 1-channel (spectrogram) input
        old_proj = vit.patch_embed.proj
        new_proj = nn.Conv2d(1, old_proj.out_channels,
                             kernel_size=old_proj.kernel_size,
                             stride=old_proj.stride)
        with torch.no_grad():
            new_proj.weight.copy_(old_proj.weight.mean(dim=1, keepdim=True))
            if old_proj.bias is not None:
                new_proj.bias.copy_(old_proj.bias)
        vit.patch_embed.proj = new_proj
        # Replace head
        vit.head = nn.Linear(vit.head.in_features, num_classes)
        self._vit = vit

    def fc(self, x: torch.Tensor) -> torch.Tensor:
        return self._vit.head(x)

    def forward_encoder(self, x: torch.Tensor):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = x.float()
        # Resize to fixed 224×224 regardless of input dimensions
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        features = self._vit.forward_features(x)  # (B, num_tokens, 384)
        cls_token = features[:, 0]  # (B, 384)
        return cls_token, []

    def forward_head(self, feature_vector: torch.Tensor) -> torch.Tensor:
        return self._vit.head(feature_vector)

    def forward(self, x: torch.Tensor):
        cls_token, feature_maps = self.forward_encoder(x)
        logits = self.forward_head(cls_token)
        return logits, cls_token, feature_maps


class ImageNet(nn.Module):
    """ImageNet — unified image-modality wrapper for CREMAD (resnet18, resnet50, vit_b_16)."""

    def __init__(self, args):
        super(ImageNet, self).__init__()
        self.arch = args.image_arch
        if self.arch == 'vit_b_16':
            self.backbone = ViTImageNet(num_classes=_N_CLASSES)
            self._feature_dim = 768
        else:
            if self.arch == 'resnet18':
                layers = [2, 2, 2, 2]
            elif self.arch == 'resnet50':
                layers = [3, 4, 6, 3]
            else:
                raise ValueError(f'Unknown image_arch: {self.arch}')
            self.backbone = _ResNetImageNet(layers, num_classes=_N_CLASSES)
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


class AudioNet(nn.Module):
    """AudioNet — supports resnet18, resnet50, vit_b_16, vit_s_16."""
    def __init__(self, args):
        super(AudioNet, self).__init__()
        self.args = args
        self.arch = args.audio_arch

        if self.arch == 'vit_b_16':
            self._vit_net = ViTAudioNet(num_classes=6, num_frame=getattr(args, 'num_frame', 1))
        elif self.arch == 'vit_s_16':
            self._vit_net = ViTSAudioNet(num_classes=6)
        else:
            if args.audio_arch == 'resnet18':
                layers = [2, 2, 2, 2]
            elif args.audio_arch == 'resnet50':
                layers = [3, 4, 6, 3]
            else:
                layers = [2, 2, 2, 2]
            self.backbone = _resnet('resnet_x', BasicBlock, layers, modality='audio',
                                    num_classes=6)
            self.head_audio = nn.Linear(512, 6)

    def _is_vit(self):
        return self.arch in ('vit_b_16', 'vit_s_16')

    def forward(self, x):
        if self._is_vit():
            return self._vit_net(x)
        bb = self.backbone
        z = bb.maxpool(bb.relu(bb.bn1(bb.conv1(x))))
        f1 = bb.layer1(z)
        f2 = bb.layer2(f1)
        f3 = bb.layer3(f2)
        f4 = bb.layer4(f3)
        pooled = F.adaptive_avg_pool2d(f4, 1)
        features = torch.flatten(pooled, 1)
        logits = self.head_audio(features)
        return logits, features, [f1, f2, f3, f4]

    def fc(self, feature_vector):
        if self._is_vit():
            return self._vit_net.fc(feature_vector)
        return self.head_audio(feature_vector)

    def forward_head(self, feature_vector):
        if self._is_vit():
            return self._vit_net.forward_head(feature_vector)
        return self.head_audio(feature_vector)

    def forward_encoder(self, x):
        if self._is_vit():
            return self._vit_net.forward_encoder(x)
        bb = self.backbone
        z = bb.maxpool(bb.relu(bb.bn1(bb.conv1(x))))
        f1 = bb.layer1(z)
        f2 = bb.layer2(f1)
        f3 = bb.layer3(f2)
        f4 = bb.layer4(f3)
        pooled = F.adaptive_avg_pool2d(f4, 1)
        feature_vector = torch.flatten(pooled, 1)
        return feature_vector, [f1, f2, f3, f4]
