import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class BidirectionalCrossAttention(nn.Module):

    def __init__(self, dim: int = 512, num_heads: int = 8):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.d_k = dim // num_heads

        self.q1 = nn.Linear(dim, dim)
        self.k2 = nn.Linear(dim, dim)
        self.v2 = nn.Linear(dim, dim)
        self.q2 = nn.Linear(dim, dim)
        self.k1 = nn.Linear(dim, dim)
        self.v1 = nn.Linear(dim, dim)
        self.out1 = nn.Linear(dim, dim)
        self.out2 = nn.Linear(dim, dim)

    def forward(self, x_eff: torch.Tensor, x_dense: torch.Tensor) -> tuple:
        B, N, dim = x_eff.shape

        Q1 = self.q1(x_eff).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        K2 = self.k2(x_dense).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        V2 = self.v2(x_dense).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        a1 = F.scaled_dot_product_attention(Q1, K2, V2, scale=self.d_k ** -0.5)
        a1 = a1.transpose(1, 2).reshape(B, N, dim)
        a1 = self.out1(a1)

        Q2 = self.q2(x_dense).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        K1 = self.k1(x_eff).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        V1 = self.v1(x_eff).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        a2 = F.scaled_dot_product_attention(Q2, K1, V1, scale=self.d_k ** -0.5)
        a2 = a2.transpose(1, 2).reshape(B, N, dim)
        a2 = self.out2(a2)

        return a1, a2


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
        sp = F.sigmoid(self.spatial_conv(torch.cat([mean_p, max_p], dim=1)))
        return x * ch * sp


class EDXAGN(nn.Module):

    def __init__(self, num_classes: int = 4, latent_dim: int = 512,
                 num_heads: int = 8, mc_passes: int = 20, dropout_rate: float = 0.3):
        super().__init__()
        self.num_classes = num_classes
        self.mc_passes = mc_passes
        self.dropout_rate = dropout_rate

        from torchvision.models import EfficientNet_B0_Weights
        effnet = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.eff_mid = nn.Sequential(*effnet.features[:6])
        self.eff_final = nn.Sequential(*effnet.features[6:])


        from torchvision.models import DenseNet121_Weights
        densenet = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        self.dense_features = densenet.features

        self.proj_eff = nn.Conv2d(1280, latent_dim, 1)
        self.proj_dense = nn.Conv2d(1024, latent_dim, 1)

        self.cross_attn = BidirectionalCrossAttention(dim=latent_dim, num_heads=num_heads)
        self.alpha = nn.Parameter(torch.zeros(1))
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
            pooled = F.avg_pool2d(feat_detached, (H, W)).view(B, -1)
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
        feat_stage4 = self.eff_mid(x)         
        feat_final = self.eff_final(feat_stage4) 

        feat_dense = self.dense_features(x)   

        e_eff = self.proj_eff(feat_final)    
        e_dense = self.proj_dense(feat_dense) 

        B, C, H, W = e_eff.shape
        x_eff = e_eff.flatten(2).transpose(1, 2)
        x_dense = e_dense.flatten(2).transpose(1, 2)

        a1, a2 = self.cross_attn(x_eff, x_dense)

        a1 = a1.reshape(B, H, W, C).permute(0, 3, 1, 2)
        a2 = a2.reshape(B, H, W, C).permute(0, 3, 1, 2)

        w = torch.sigmoid(self.alpha)
        f_gated = w * a1 + (1 - w) * a2
        f_refined = self.asca(f_gated)

        z = F.avg_pool2d(f_refined, f_refined.shape[-1]).view(B, -1)
        cls_logits = self.class_head(z)

        if compute_uncertainty:
            z_rep = z.unsqueeze(0).repeat(self.mc_passes, 1, 1)
            z_drop = F.dropout(z_rep, p=self.dropout_rate, training=True)
            mc_logits = self.class_head(z_drop)
            mc_probs = F.softmax(mc_logits, dim=-1)
            p_var = mc_probs.var(dim=0).mean(dim=1)
        else:
            p_var = torch.zeros(B, device=x.device)

        f_early = self.gradcam_reduce(feat_stage4)
        aux_logits = self.aux_classifier(
            F.avg_pool2d(f_early, (14, 14)).view(B, -1)
        )
        if return_heatmap:
            heatmap = self._gradcam_heatmap(f_early)
            heatmap = F.interpolate(heatmap, size=224, mode='bilinear', align_corners=False)
        else:
            heatmap = None

        return {
            'cls_logits': cls_logits,
            'cls_probs': F.softmax(cls_logits, dim=1),
            'uncertainty': p_var,
            'aux_logits': aux_logits,
            'heatmap': heatmap,
        }
