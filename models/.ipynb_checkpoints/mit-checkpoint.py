import torch
from torch import nn
from collections import OrderedDict
from timm.models.layers import trunc_normal_
import sys
sys.path.append("../")
from clip.model import QuickGELU


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.n_head = n_head

    def attention(self, x, attn_mask=None):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        attn_mask = attn_mask.to(dtype=x.dtype, device=x.device) if attn_mask is not None else self.attn_mask
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x):
        x, sample_range = x
        # add by L10
        # construct attn_mask
        # 1 means masked; 0 means unmasked
        L, N, E = x.shape
        S = L
        attn_mask = torch.ones((N * self.n_head, L, S))
        for i in range(len(sample_range)):
            attn_mask[i * self.n_head: (i+1) * self.n_head, :sample_range[i], :sample_range[i]] = 0
        ###
        x = x + self.attention(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class MultiframeIntegrationTransformer(nn.Module):
    def __init__(self, T, embed_dim=512, layers=1,):
        super().__init__()
        self.T = T
        transformer_heads = embed_dim // 64
        self.positional_embedding = nn.Parameter(torch.empty(1, T, embed_dim))
        trunc_normal_(self.positional_embedding, std=0.02)
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(d_model=embed_dim, n_head=transformer_heads) for _ in range(layers)])

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear,)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x, sample_range):
        ori_x = x
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.resblocks([x, sample_range])
        x = x.permute(1, 0, 2)  
        x = x.type(ori_x.dtype) + ori_x

        # add by L10
        ### cal for mask
        x_new = []
        for i in range(len(sample_range)):
            x_new.append(x[i, :sample_range[i]].sum(dim=0) / sample_range[i])
        x_new = torch.stack(x_new)
        # x = x.mean(dim=1, keepdim=False)
        return x_new
