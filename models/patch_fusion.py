import torch
from torch import nn
from collections import OrderedDict
from timm.models.layers import trunc_normal_
import sys
sys.path.append("../")
from clip.model import QuickGELU
import torch.nn.functional as F
import math

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.ln_11 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.n_head = n_head

    def attention(self, x, y, attn_mask=None):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        attn_mask = attn_mask.to(dtype=x.dtype, device=x.device) if attn_mask is not None else self.attn_mask
        return self.attn(x, y, y, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x):
        x, prompt, sample_range = x
        if prompt is None:
            # S, N, E = x.shape
            # L = S
            # attn_mask = torch.empty((N * self.n_head, L, S)).fill_(float("-inf"))
            # for i in range(len(sample_range)):
            #     attn_mask[i * self.n_head: (i + 1) * self.n_head, :, :sample_range[i]].fill_(0.0)
            attn_mask = None
            x = x + self.attention(self.ln_1(x), self.ln_1(x), attn_mask)
            x = x + self.mlp(self.ln_2(x))
        else:
            S, N, E = x.shape
            L = prompt.shape[0]
            prompt_repeat = prompt.repeat(N, 1, 1).permute(1, 0, 2)
            attn_mask = torch.empty((N * self.n_head, L, S)).fill_(float("-inf"))
            for i in range(len(sample_range)):
                attn_mask[i * self.n_head: (i+1) * self.n_head, :, :sample_range[i]].fill_(0.0)
            x = self.attention(self.ln_11(prompt_repeat), self.ln_1(x), attn_mask)
            x = self.mlp(self.ln_2(x))
        return [x, prompt, sample_range]

class MultiframeIntegrationTransformer(nn.Module):
    def __init__(self, T, embed_dim=512, layers=1,):
        super().__init__()
        self.T = T
        transformer_heads = embed_dim // 64
        self.positional_embedding = nn.Parameter(torch.empty(1, T, embed_dim))
        trunc_normal_(self.positional_embedding, std=0.02)
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(d_model=embed_dim, n_head=transformer_heads) for _ in range(layers)])
        self.pad_embed = nn.Embedding(1, embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear,)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x, prompt, patch_info):
        # unpack patch info data
        sample_range = patch_info['sample_range']
        posis_rlt, posis_enc = get_pos_embed(patch_info, x.shape[-1])
        # posx_rlt, posy_rlt, posx_enc, posy_enc = get_pos_embed(patch_info, x.shape[-1])

        # pad x data
        x_new = self.pad_embed(torch.zeros(x.shape[0], x.shape[1], dtype=torch.long, device=x.device))
        for i in range(len(sample_range)):
            x_new[i, :sample_range[i]] = x[i, :sample_range[i]]
        x = x_new

        # attn process
        ori_x = x
        # if prompt is None:
        #     x = x + posx_enc.to(x.device) + posy_enc.to(x.device)
        x = x + posis_enc.to(x.device)
        # x = x + posx_enc.to(x.device) + posy_enc.to(x.device)
        # x = x + self.positional_embedding
        x = x.permute(1, 0, 2)
        x, _, _ = self.resblocks([x, prompt, sample_range])
        x = x.permute(1, 0, 2)

        if prompt is None:
            x = x.type(ori_x.dtype) + ori_x
        return x

class PatchFusionTransformer(nn.Module):
    def __init__(self, T, embed_dim=512, layers_sa=1, layers_ca=1):
        super().__init__()
        self.self_attn = MultiframeIntegrationTransformer(T, embed_dim, layers_sa)
        self.cross_attn = MultiframeIntegrationTransformer(T, embed_dim, layers_ca)
        self.bn1 = nn.LayerNorm(embed_dim)
        self.bn2 = nn.LayerNorm(embed_dim)
        # self.project_for_cls = nn.Sequential(OrderedDict([
        #     ('fc1', nn.Linear(embed_dim, embed_dim//16)),
        #     ('relu1', nn.LeakyReLU()),
        #     ('fc2', nn.Linear(embed_dim/16, embed_dim)),
        # ]))
        # self.project_for_cls = nn.Identity()
        self.project_for_cls = nn.Linear(embed_dim * 2, embed_dim)
        # self.project_for_cls = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, prompt, patch_info):
        sample_range = patch_info['sample_range']

        x = self.self_attn(x, None, patch_info)
        x_new = []
        for i in range(len(sample_range)):
            x_new.append(x[i, :sample_range[i]].sum(dim=0) / sample_range[i])
        x_new = torch.stack(x_new)

        x = self.cross_attn(x, prompt, patch_info)
        x = x.mean(dim=1, keepdim=False)

        # whether to fusion
        x = self.project_for_cls(torch.cat([x, x_new], dim=1))
        # x = self.project_for_cls(x + x_new)
        # x = self.project_for_cls(self.bn1(x) + self.bn2(x_new))
        return x

def get_pos_embed(patch_info, d_model):
    posis = patch_info['patch_inds']
    patch_pub_cnt = patch_info['patch_pub_cnt']
    ###
    posis_rlt = posis / patch_pub_cnt.reshape(-1, 1)
    posis_enc = torch.zeros((posis_rlt.shape[0], posis_rlt.shape[1], d_model))

    for i in range(d_model):
        if i % 2 == 0:
            posis_enc[..., i] = torch.sin(posis_rlt / (10000 ** (2 * i / d_model)))
        else:
            posis_enc[..., i] = torch.cos(posis_rlt / (10000 ** ((2 * i - 1) / d_model)))

    return posis_rlt, posis_enc

# def get_pos_embed(patch_info, d_model):
#     posis = patch_info['posis']
#     patch_maxr = patch_info['patch_maxr']
#     patch_maxc = patch_info['patch_maxc']
#     ###
#     posx_rlt = posis[..., 0] / patch_maxr.reshape(-1, 1)
#     posy_rlt = posis[..., 1] / patch_maxc.reshape(-1, 1)
# 
#     posx_enc = torch.zeros((posx_rlt.shape[0], posx_rlt.shape[1], d_model))
#     posy_enc = torch.zeros((posy_rlt.shape[0], posy_rlt.shape[1], d_model))
# 
#     for i in range(d_model):
#         if i % 2 == 0:
#             posx_enc[..., i] = torch.sin(posx_rlt / (10000 ** (2 * i / d_model)))
#             posy_enc[..., i] = torch.sin(posy_rlt / (10000 ** (2 * i / d_model)))
#         else:
#             posx_enc[..., i] = torch.cos(posx_rlt / (10000 ** ((2 * i - 1) / d_model)))
#             posy_enc[..., i] = torch.cos(posy_rlt / (10000 ** ((2 * i - 1) / d_model)))
# 
#     return posx_rlt, posy_rlt, posx_enc, posy_enc

