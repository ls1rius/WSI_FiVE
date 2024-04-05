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
        # attn_mask = attn_mask.to(dtype=bool, device=x.device) if attn_mask is not None else self.attn_mask
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x):
        x, sample_range = x
        # # add by L10
        # # construct attn_mask
        # # 1 means masked; 0 means unmasked
        # L, N, E = x.shape
        # attn_mask = torch.empty((N * self.n_head, L, L)).fill_(float("-inf"))
        # for i in range(len(sample_range)):
        #     # attn_mask[i * self.n_head: (i+1) * self.n_head, :sample_range[i], :sample_range[i]].triu_(1)
        #     attn_mask[i * self.n_head: (i+1) * self.n_head, :sample_range[i], :sample_range[i]].fill_(0.0)
        # # ###
        attn_mask = None
        x = x + self.attention(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return [x, sample_range]

class MultiframeIntegrationTransformer(nn.Module):
    def __init__(self, T, embed_dim=512, layers=1,):
        super().__init__()
        self.recur_layers = 2
        self.T = T
        transformer_heads = embed_dim // 64

        # self.positional_embedding = nn.Parameter(torch.empty(self.recur_layers, 1, T, embed_dim))
        self.positional_embedding = nn.Parameter(torch.empty(1, T, embed_dim))
        trunc_normal_(self.positional_embedding, std=0.02)
        # self.resblocks = nn.ModuleList([nn.Sequential(*[ResidualAttentionBlock(d_model=embed_dim, n_head=transformer_heads) for _ in range(layers)]) for __ in range(self.recur_layers)])
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(d_model=embed_dim, n_head=transformer_heads) for _ in range(layers)])
        # self.pad_embed = nn.Embedding(self.recur_layers, embed_dim)
        self.pad_embed = nn.Embedding(1, embed_dim)

        # self.fc1 = nn.Linear(embed_dim, embed_dim // 16)
        # self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(embed_dim // 16, embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear,)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    # def attn_dfs(self, x, patch_info):
    #     # unpack patch info data
    #     sample_range = patch_info['sample_range']
    #     grad_batch = patch_info['GRAD_BATCH']
    #     num_frames = patch_info['NUM_FRAMES']
    #     # posx_rlt, posy_rlt, posx_enc, posy_enc = self.get_pos_embed(patch_info, x.shape[-1])
    # 
    #     assert x.shape[1] == num_frames
    #     for ind in range(self.recur_layers):
    #         B, N, C = x.shape
    #         N_attn = math.ceil(N/grad_batch) # batch attention num with ceil deal
    #         # pad x data
    #         x_new = self.pad_embed(torch.ones(B, N_attn*grad_batch, dtype=torch.long, device=x.device) * ind)
    #         for i in range(len(sample_range)):
    #             x_new[i, :sample_range[i]] = x[i, :sample_range[i]]
    #         x = x_new.view(B*N_attn, grad_batch, C)
    #         ###
    #         # cal attn batch sample range
    #         x_len_range = torch.arange(B*N_attn, device=x.device) % N_attn * grad_batch
    #         x_sample_max = sample_range.reshape(-1, 1).repeat(1, N_attn).reshape(-1)
    #         x_sample_range = torch.clamp(x_sample_max-x_len_range, 0, grad_batch)
    #         ###
    #         # attn process
    #         ori_x = x
    #         x = x + self.positional_embedding[ind]
    #         x = x.permute(1, 0, 2)
    #         x, _ = self.resblocks[ind]([x, x_sample_range])
    #         x = x.permute(1, 0, 2)
    #         x = x.type(ori_x.dtype) + ori_x
    #         ###
    #         # cal for mask
    #         x_new = []
    #         for i in range(len(x_sample_range)):
    #             x_new.append(x[i, :x_sample_range[i]].sum(dim=0) / x_sample_range[i])
    #         x = torch.stack(x_new).view(B, N_attn, C)
    #         ###
    #         sample_range = (sample_range/grad_batch).ceil().long()
    #     assert N_attn == 1
    #     x = x.squeeze(dim=1)
    #     return x
    # 
    # def forward(self, x, patch_info):
    #     x = self.attn_dfs(x, patch_info)
    #     # x = self.fc2(self.relu(self.fc1(x)))
    #     return x

    def forward(self, x, prompt, patch_info):
        # unpack patch info data
        sample_range = patch_info['sample_range']
        grad_batch = patch_info['GRAD_BATCH']
        num_frames = patch_info['NUM_FRAMES']
        posx_rlt, posy_rlt, posx_enc, posy_enc = self.get_pos_embed(patch_info, x.shape[-1])

        # pad x data
        x_new = self.pad_embed(torch.zeros(x.shape[0], x.shape[1], dtype=torch.long, device=x.device))
        for i in range(len(sample_range)):
            x_new[i, :sample_range[i]] = x[i, :sample_range[i]]
        x = x_new
        ###

        # attn process
        ori_x = x
        # x = x + F.normalize(posx_rlt, p=2, dim=-1).unsqueeze(dim=-1).to(x.device) + \
        #     F.normalize(posy_rlt, p=2, dim=-1).unsqueeze(dim=-1).to(x.device)
        # x = x + posx_enc.to(x.device) + posy_enc.to(x.device)
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)
        x, _ = self.resblocks([x, sample_range])
        x = x.permute(1, 0, 2)
        x = x.type(ori_x.dtype) + ori_x
        ###

        # add by L10
        ### cal for mask
        x_new = []
        for i in range(len(sample_range)):
            x_new.append(x[i, :sample_range[i]].sum(dim=0) / sample_range[i])
        x = torch.stack(x_new)
        ###
        # x = x.mean(dim=1, keepdim=False)
        return x

    def get_pos_embed(self, patch_info, d_model):
        posis = patch_info['posis']
        patch_maxr = patch_info['patch_maxr']
        patch_maxc = patch_info['patch_maxc']
        ###
        posx_rlt = posis[..., 0] / patch_maxr.reshape(-1, 1)
        posy_rlt = posis[..., 1] / patch_maxc.reshape(-1, 1)

        posx_enc = torch.zeros((posx_rlt.shape[0], posx_rlt.shape[1], d_model))
        posy_enc = torch.zeros((posy_rlt.shape[0], posy_rlt.shape[1], d_model))

        for i in range(d_model):
            if i % 2 == 0:
                posx_enc[..., i] = torch.sin(posx_rlt / (10000 ** (2 * i / d_model)))
                posy_enc[..., i] = torch.sin(posy_rlt / (10000 ** (2 * i / d_model)))
            else:
                posx_enc[..., i] = torch.cos(posx_rlt / (10000 ** ((2 * i - 1) / d_model)))
                posy_enc[..., i] = torch.cos(posy_rlt / (10000 ** ((2 * i - 1) / d_model)))

        return posx_rlt, posy_rlt, posx_enc, posy_enc
