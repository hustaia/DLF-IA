import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_
import math


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=None):
        super().__init__()

        self.dim = dim
        self.scale = dim ** -0.5
        
        self.q = nn.Linear(2 * dim, dim)
        self.k1 = nn.Linear(dim, dim)
        self.k2 = nn.Linear(dim, dim)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, x1, x2):
        B, N, C = x1.shape
        x_cat = torch.cat([x1, x2], -1)
        q = self.q(x_cat)
        k1 = self.k1(x1)
        k2 = self.k2(x2)
        a1 = torch.multiply(q, k1) * self.scale
        a2 = torch.multiply(q, k2) * self.scale
        return a1, a2
        
        
class FeatureFusionModule(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None):
        super().__init__()
        self.channel_proj1 = nn.Linear(dim, dim // reduction)
        self.channel_proj2 = nn.Linear(dim, dim // reduction)
        self.act1 = nn.ReLU(inplace=True)
        self.cross_attn = CrossAttention(dim // reduction, num_heads=num_heads)
        self.end_proj1 = nn.Linear(dim // reduction, dim)
        self.end_proj2 = nn.Linear(dim // reduction, dim)
        self.act2 = nn.Sigmoid()

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        u1 = self.act1(self.channel_proj1(x1))
        u2 = self.act1(self.channel_proj2(x2))
        a1, a2 = self.cross_attn(u1, u2)
        v1 = self.act2(self.end_proj1(a1))
        v2 = self.act2(self.end_proj2(a2))
        y1 = v1.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        y2 = v2.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        return y1, y2
