"""
Shared KD loss modules for UCMKD.

ReviewKDLoss  — "Distilling Knowledge via Knowledge Review" (Chen et al., CVPR 2021)
               Cross-level distillation with Attention-Based Fusion (ABF) + Hierarchical
               Context Loss (HCL). Shallower teacher stages supervise deeper student stages
               via learned attention-weighted residual fusion.

NORMLoss      — "Knowledge Distillation via N-to-One Representation Matching" (ICLR 2023)
               Expands student features N× through a learnable FT module, then matches
               the expanded student against the teacher feature tiled N times (MSE).
               Zero inference overhead — FT layers fold into the subsequent FC layer.

Both modules use lazy initialization: ABF/FT parameters are created on the first forward
pass (so channel dims are inferred from actual tensors). Each module owns its own
internal Adam optimizer; call .zero_grad_kd() before the main backward and .step_kd()
after the main optimizer step.

Usage example (inside a training loop):
    review_fn = ReviewKDLoss(lr=1e-3)
    norm_fn   = NORMLoss(N=4, lr=1e-3)
    ...
    review_fn.zero_grad_kd(); norm_fn.zero_grad_kd()
    optimizer.zero_grad()
    ...
    review_loss = review_fn(stu_fit, tea_fit)   # stu_fit, tea_fit: list of 5 feature maps
    norm_loss   = norm_fn(stu_f, tea_f)         # stu_f, tea_f: [B, D] pooled vectors
    loss = ce_loss + fa_w * fa_loss + la_w * la_loss \
         + args.review_weight * review_loss + args.norm_weight * norm_loss
    loss.backward()
    optimizer.step(); review_fn.step_kd(); norm_fn.step_kd()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# ReviewKD: ABF + HCL
# ─────────────────────────────────────────────────────────────────────────────

class ABF(nn.Module):
    """Attention-Based Fusion module.

    Projects student features to teacher channel space (conv1 → mid_ch → conv2 → tea_ch).
    When a 'residual' from the deeper stage is provided, uses learned attention weights
    to blend the current-level projection with the upsampled residual before the final
    projection. This implements the progressive cross-level review mechanism.

    mid_ch is shared across all ABF modules in a ReviewKDLoss instance so that the
    residual tensor passed between levels always has a consistent channel dimension.
    """

    def __init__(self, stu_ch: int, tea_ch: int, mid_ch: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(stu_ch, mid_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_ch),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_ch, tea_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(tea_ch),
        )
        # attention: blend current mid-feat with upsampled residual (both mid_ch)
        self.att = nn.Sequential(
            nn.Conv2d(mid_ch * 2, 2, kernel_size=1),
            nn.Sigmoid(),
        )
        self._mid_ch = mid_ch
        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)

    def forward(self, x: torch.Tensor, residual=None):
        """
        x        : student feature map  [B, stu_ch, H, W]
        residual : mid-feat from deeper stage [B, mid_ch, H', W'] or None (deepest level)
        Returns  : (out [B, tea_ch, H, W], mid [B, mid_ch, H, W])
                   'mid' is passed as 'residual' to the next (shallower) ABF call.
        """
        mid = self.conv1(x)                      # [B, mid_ch, H, W]
        if residual is not None:
            res_up = F.interpolate(residual, size=mid.shape[2:], mode='nearest')
            attn = self.att(torch.cat([mid, res_up], dim=1))  # [B, 2, H, W]
            mid  = attn[:, 0:1] * mid + attn[:, 1:2] * res_up
        out = self.conv2(mid)                    # [B, tea_ch, H, W]
        return out, mid


def hcl_loss(fs: torch.Tensor, ft: torch.Tensor) -> torch.Tensor:
    """Hierarchical Context Loss (HCL).

    Multi-scale spatial pyramid MSE: compares full resolution plus three pooled
    versions (4×4, 2×2, 1×1).  Contributions are halved at each coarser scale
    (following the paper's weighting scheme).

    fs, ft : [B, C, H, W]  — must have identical shape.
    """
    _, _, h, w = fs.shape
    loss = F.mse_loss(fs, ft)
    cnt, tot = 1.0, 1.0
    for scale in [4, 2, 1]:
        if scale >= h:          # skip if spatial size already smaller
            continue
        cnt /= 2.0
        loss = loss + F.mse_loss(
            F.adaptive_avg_pool2d(fs, (scale, scale)),
            F.adaptive_avg_pool2d(ft, (scale, scale)),
        ) * cnt
        tot += cnt
    return loss / tot


class ReviewKDLoss(nn.Module):
    """ReviewKD full loss: ABF-based progressive fusion + HCL at each stage.

    Lazy-initialises ABF modules and an internal Adam optimizer on the first
    forward call (channel dims are inferred from input tensors).

    Protocol each training iteration:
        review_fn.zero_grad_kd()
        ...compute loss, call loss.backward()...
        review_fn.step_kd()
    """

    def __init__(self, lr: float = 1e-3, weight_decay: float = 1e-4):
        super().__init__()
        self._lr = lr
        self._wd = weight_decay
        self.abfs: nn.ModuleList | None = None
        self._optim = None
        self._tea_hw = None

    # ------------------------------------------------------------------
    def _lazy_init(self, stu_feats, tea_feats):
        n = len(stu_feats)
        stu_chs = [f.shape[1] for f in stu_feats]
        tea_chs = [f.shape[1] for f in tea_feats]
        # Use teacher's last feature map spatial size as target
        self._tea_hw = (tea_feats[-1].shape[2], tea_feats[-1].shape[3])
        device = stu_feats[0].device
        # Shared mid_ch across all ABFs so residual channels are consistent
        mid_ch = min(min(stu_chs), min(tea_chs), 256)

        self.abfs = nn.ModuleList(
            [ABF(stu_chs[i], tea_chs[i], mid_ch) for i in range(n)]
        ).to(device)

        self._optim = torch.optim.Adam(
            self.abfs.parameters(), lr=self._lr, weight_decay=self._wd
        )

    # ------------------------------------------------------------------
    def zero_grad_kd(self):
        if self._optim is not None:
            self._optim.zero_grad()

    def step_kd(self):
        if self._optim is not None:
            self._optim.step()

    # ------------------------------------------------------------------
    def forward(self, stu_feats, tea_feats) -> torch.Tensor:
        """
        stu_feats : list of 5 student feature maps [f0..f4], shallow→deep
        tea_feats : list of 5 teacher feature maps [f0..f4], shallow→deep
        Returns   : scalar ReviewKD loss
        """
        if self.abfs is None:
            self._lazy_init(stu_feats, tea_feats)

        th, tw = self._tea_hw
        n = len(stu_feats)

        # ── Progressive fusion: deep → shallow ──────────────────────────
        residual = None
        proj_results = []
        for i in reversed(range(n)):
            out, residual = self.abfs[i](stu_feats[i], residual)
            # Resize fused output to teacher spatial resolution
            if out.shape[2:] != (th, tw):
                out = F.interpolate(out, size=(th, tw), mode='bilinear', align_corners=False)
            if residual.shape[2:] != (th, tw):
                residual = F.interpolate(residual, size=(th, tw), mode='bilinear', align_corners=False)
            proj_results.insert(0, out)

        # ── HCL at each level ───────────────────────────────────────────
        total = torch.tensor(0.0, device=stu_feats[0].device)
        for fs, ft in zip(proj_results, tea_feats):
            ft_d = ft.detach()
            if ft_d.shape[2:] != fs.shape[2:]:
                ft_d = F.interpolate(ft_d, size=fs.shape[2:], mode='bilinear', align_corners=False)
            total = total + hcl_loss(fs, ft_d)

        return total / n


# ─────────────────────────────────────────────────────────────────────────────
# NORM: N-to-One Representation Matching
# ─────────────────────────────────────────────────────────────────────────────

class NORMLoss(nn.Module):
    """NORM loss: expands student feature to N × tea_dim via a learnable FT module,
    then matches against teacher feature tiled N times.

    FT module: expand (stu_dim → tea_dim×N) + contract (tea_dim×N → stu_dim) with
    residual — same structure as in the original paper.  The contract branch keeps the
    student backbone's representation intact; only the expand branch feeds into the loss.

    Lazy-initialises FT and an internal Adam optimizer on the first forward call.

    Protocol each training iteration:
        norm_fn.zero_grad_kd()
        ...compute loss, call loss.backward()...
        norm_fn.step_kd()
    """

    def __init__(self, N: int = 4, lr: float = 1e-3, weight_decay: float = 1e-4):
        super().__init__()
        self.N = N
        self._lr = lr
        self._wd = weight_decay
        self.ft_expand   = None   # Linear(stu_dim, tea_dim * N)
        self.ft_contract = None   # Linear(tea_dim * N, stu_dim)  [residual branch]
        self._optim = None
        self._initialized = False

    # ------------------------------------------------------------------
    def _lazy_init(self, stu_feat: torch.Tensor, tea_feat: torch.Tensor):
        stu_dim = stu_feat.shape[-1]
        tea_dim = tea_feat.shape[-1]
        expanded = tea_dim * self.N
        device   = stu_feat.device

        self.ft_expand   = nn.Linear(stu_dim, expanded,  bias=False).to(device)
        self.ft_contract = nn.Linear(expanded, stu_dim,  bias=False).to(device)
        nn.init.kaiming_normal_(self.ft_expand.weight)
        nn.init.kaiming_normal_(self.ft_contract.weight)

        self._optim = torch.optim.Adam(
            list(self.ft_expand.parameters()) + list(self.ft_contract.parameters()),
            lr=self._lr, weight_decay=self._wd,
        )
        self._initialized = True

    # ------------------------------------------------------------------
    def zero_grad_kd(self):
        if self._optim is not None:
            self._optim.zero_grad()

    def step_kd(self):
        if self._optim is not None:
            self._optim.step()

    # ------------------------------------------------------------------
    def forward(self, stu_feat: torch.Tensor, tea_feat: torch.Tensor) -> torch.Tensor:
        """
        stu_feat : [B, stu_dim] student pooled feature (or [B, C, H, W] — GAP applied)
        tea_feat : [B, tea_dim] teacher pooled feature (or [B, C, H, W] — GAP applied)
        Returns  : scalar NORM loss
        """
        # GAP if spatial feature maps are passed
        if stu_feat.dim() == 4:
            stu_feat = F.adaptive_avg_pool2d(stu_feat, 1).flatten(1)
        if tea_feat.dim() == 4:
            tea_feat = F.adaptive_avg_pool2d(tea_feat, 1).flatten(1)

        stu_feat = stu_feat.float()
        tea_feat = tea_feat.float()

        if not self._initialized:
            self._lazy_init(stu_feat, tea_feat)

        # Expand student: [B, stu_dim] → [B, tea_dim * N]
        stu_expanded = self.ft_expand(stu_feat)

        # Tile teacher: [B, tea_dim] → [B, tea_dim * N]
        tea_tiled = tea_feat.detach().repeat(1, self.N)

        # NORM loss = N × MSE  (N factor normalises for the N repetitions)
        return F.mse_loss(stu_expanded, tea_tiled) * self.N
