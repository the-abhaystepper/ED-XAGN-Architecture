import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from src.utils.gamma import gamma_correction, imagenet_normalize


class ResidualSelfAttention(nn.Module):
    """Multi-head self-attention with a residual back to the CNN tokens.

    The output projection is zero-initialized so the block starts as the identity
    and the pretrained EfficientNet map is not replaced at step zero.
    """

    def __init__(self, dim: int = 512, num_heads: int = 8):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.d_k = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, dim = x.shape
        Q = self.q(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        attn = F.scaled_dot_product_attention(Q, K, V, scale=self.d_k ** -0.5)
        attn = attn.transpose(1, 2).reshape(B, N, dim)
        return x + self.out(attn)


class ASCA(nn.Module):

    def __init__(self, dim: int = 512, reduction: int = 16):
        super().__init__()
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1),
            nn.Sigmoid(),
        )
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ch = self.channel_attn(x)
        mean_p = x.mean(dim=1, keepdim=True)
        max_p = x.max(dim=1, keepdim=True)[0]
        sp = torch.sigmoid(self.spatial_conv(torch.cat([mean_p, max_p], dim=1)))
        return x * ch * sp


class EDXAGN(nn.Module):

    def __init__(self, num_classes: int = 3, latent_dim: int = 512,
                 num_heads: int = 8, mc_passes: int = 20, dropout_rate: float = 0.3,
                 gamma: float = 1.2, gamma_scale: float = 1.0, heatmap_size: int = 224):
        super().__init__()
        self.num_classes = num_classes
        self.mc_passes = mc_passes
        self.dropout_rate = dropout_rate
        self.gamma = gamma
        self.gamma_scale = gamma_scale
        self.heatmap_size = heatmap_size

        from torchvision.models import EfficientNet_B0_Weights
        effnet = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1, progress=False)
        # features[:6] ends at stage 5: 112 channels, 14x14 on a 224 input.
        self.eff_mid = nn.Sequential(*effnet.features[:6])
        self.eff_final = nn.Sequential(*effnet.features[6:])

        self.proj = nn.Conv2d(1280, latent_dim, 1)
        self.self_attn = ResidualSelfAttention(dim=latent_dim, num_heads=num_heads)
        self.asca = ASCA(dim=latent_dim)
        self.class_head = nn.Linear(latent_dim, num_classes)
        self.uncert_dropout = nn.Dropout(p=dropout_rate)
        self.uncert_head = nn.Linear(latent_dim, num_classes)

        self.gradcam_reduce = nn.Sequential(
            nn.Conv2d(112, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.aux_classifier = nn.Linear(64, num_classes)

    def _gradcam_heatmap(self, features: torch.Tensor, class_idx: torch.Tensor = None) -> torch.Tensor:
        B, C, H, W = features.shape
        feat_detached = features.detach().requires_grad_(True)

        with torch.enable_grad():
            pooled = F.adaptive_avg_pool2d(feat_detached, 1).view(B, -1)
            aux_scores = self.aux_classifier(pooled)

            if class_idx is None:
                class_idx = aux_scores.argmax(dim=1)
            elif not isinstance(class_idx, torch.Tensor):
                class_idx = torch.tensor(class_idx, device=features.device).expand(B)

            target_scores = aux_scores.gather(1, class_idx.view(B, 1)).sum()
            grads = torch.autograd.grad(target_scores, feat_detached, retain_graph=False, create_graph=False)[0]

        alpha_k = grads.mean(dim=(2, 3), keepdim=True)
        heatmap = (alpha_k * feat_detached).sum(dim=1, keepdim=True)
        heatmap = F.relu(heatmap)
        min_val = heatmap.amin(dim=(2, 3), keepdim=True)
        max_val = heatmap.amax(dim=(2, 3), keepdim=True)
        heatmap = (heatmap - min_val) / (max_val - min_val + 1e-8)
        return heatmap

    def forward(self, x: torch.Tensor, compute_uncertainty: bool = True, return_heatmap: bool = True) -> dict:
        x = gamma_correction(x, gamma=self.gamma, A=self.gamma_scale)
        x = imagenet_normalize(x)

        feat_mid = self.eff_mid(x)
        feat_final = self.eff_final(feat_mid)

        tokens_2d = self.proj(feat_final)
        B, C, H, W = tokens_2d.shape
        tokens = tokens_2d.flatten(2).transpose(1, 2)
        tokens = self.self_attn(tokens)
        refined = tokens.reshape(B, H, W, C).permute(0, 3, 1, 2)
        refined = self.asca(refined)

        z = F.adaptive_avg_pool2d(refined, 1).view(B, -1)
        cls_logits = self.class_head(z)
        uncert_logits = self.uncert_head(self.uncert_dropout(z))

        if compute_uncertainty:
            z_rep = z.unsqueeze(0).repeat(self.mc_passes, 1, 1)
            z_drop = F.dropout(z_rep, p=self.dropout_rate, training=True)
            mc_logits = self.uncert_head(z_drop)
            mc_probs = F.softmax(mc_logits, dim=-1)
            p_var = mc_probs.var(dim=0).mean(dim=1)
            uncert_probs = mc_probs.mean(dim=0)
        else:
            p_var = torch.zeros(B, device=z.device)
            uncert_probs = F.softmax(uncert_logits, dim=1)

        f_early = self.gradcam_reduce(feat_mid)
        aux_logits = self.aux_classifier(F.adaptive_avg_pool2d(f_early, 1).view(B, -1))
        if return_heatmap:
            heatmap = self._gradcam_heatmap(f_early)
            heatmap = F.interpolate(heatmap, size=self.heatmap_size, mode="bilinear", align_corners=False)
        else:
            heatmap = None

        return {
            "cls_logits": cls_logits,
            "cls_probs": F.softmax(cls_logits, dim=1),
            "uncert_logits": uncert_logits,
            "uncert_probs": uncert_probs,
            "uncertainty": p_var,
            "aux_logits": aux_logits,
            "heatmap": heatmap,
        }
