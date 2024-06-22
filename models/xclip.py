from typing import Tuple, Union
import torch
from torch import nn
import numpy as np
import sys
import warnings
sys.path.append("../")
from clip.model import CLIP,LayerNorm,Transformer,VisionTransformer
import clip
import math
from .patch_fusion import PatchFusionTransformer

import diffdist.functional as diff_dist
import torch.distributed as dist
from collections import OrderedDict
from .MedCLIPModel import MedCLIPVisionModelViT, MedCLIPTextModel
import random
from .lora_wrap import LoraWrap
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.bert.modeling_bert import BertModel

def dist_collect(x):
    """ collect all tensor from all GPUs
    args:
        x: shape (mini_batch, ...)
    returns:
        shape (mini_batch * num_gpu, ...)
    """
    x = x.contiguous()
    out_list = [torch.zeros_like(x, device=x.device, dtype=x.dtype).contiguous() for _ in range(dist.get_world_size())]
    out_list = diff_dist.all_gather(out_list, x)
    return torch.cat(out_list, dim=0).contiguous()

class XCLIP(CLIP):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int, 
                 # video
                 T=8, 
                 droppath=0.,
                 mit_layers=1,
                 # prompt 
                 prompts_alpha=1e-4,
                 prompts_layers=1,
                 # other
                 use_cache=True,
                 use_checkpoint=False,
                 T_mit=8,
                 is_img_pth=True,
                 ):
        super().__init__(
            embed_dim,
            image_resolution, vision_layers, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
        )
        self.use_cache = use_cache
        self.mit = PatchFusionTransformer(T, embed_dim=embed_dim, layers_sa=1, layers_ca=1,)
        self.is_img_pth = is_img_pth

        if not is_img_pth:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
            )
        else:
            # self.visual = MedCLIPVisionModelViT()
            self.visual = nn.Identity()
        # self.transformer = Transformer(
        #     width=transformer_width,
        #     layers=transformer_layers,
        #     heads=transformer_heads,
        #     attn_mask=self.build_attention_mask()
        # )
        # self.vocab_size = vocab_size
        # self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        # self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        # self.ln_final = LayerNorm(transformer_width)
        # self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

        self.transformer = MedCLIPTextModel()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.cache_text_features = None
        self.cache_prompt_features = None

        # self.prompts_visual_ln = LayerNorm(vision_width)
        # self.prompts_visual_proj = nn.Parameter(torch.randn(vision_width, embed_dim))
        # self.initialize_parameters()

        # add additional by L10
        # self.prompt_learn_param = nn.Parameter(torch.empty(16, embed_dim))
        self.prompt_learn_param = nn.Parameter(torch.empty(16, context_length, 768))
        nn.init.normal_(self.prompt_learn_param, std=0.01)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'positional_embedding'}

    def encode_image(self, image, bs=None):
        return self.visual(image)

    # L10 add for prompt addition
    def encode_prompt_addition(self, prompt):
        eos_indx = prompt.shape[1] - 1
        x = prompt
        K, N1, C = x.shape
        # change by L10
        # x = x + self.positional_embedding.repeat(N1//77, 1)
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), eos_indx] @ self.text_projection
        x = x.reshape(K, -1)
        return x

    def encode_text(self, text):
        x = self.token_embedding(text)
        eos_indx = text.argmax(dim=-1)
        K, N1, C = x.shape
        # change by L10
        # x = x + self.positional_embedding.repeat(N1//77, 1)
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), eos_indx] @ self.text_projection
        x = x.reshape(K, -1)
        return x

    # def encode_video(self, image, patch_info):
    #     b,t,c,h,w = image.size()
    #     image = image.reshape(-1,c,h,w)
    #
    #     cls_features, img_features = self.encode_image(image)
    #     cls_features = cls_features.view(b, t, -1)
    #     cls_features = self.mit(cls_features)
    #
    #     return cls_features, None

    def encode_video(self, image, prompt, patch_info):
        b, t, c, h, w = image.size()
        image = image.reshape(-1, c, h, w)
        grad_batch = patch_info['GRAD_BATCH']
        assert grad_batch <= t  # 3:1
        if grad_batch > 0 and patch_info['training']:
            ### create smaple inds
            sample_inds_bfr, sample_inds_mid, sample_inds_aft = torch.zeros(b, t), torch.zeros(b, t), torch.zeros(b, t)
            sample_inds_bfr[:, :grad_batch//4*3] = 1
            sample_inds_mid[:, grad_batch//4*3: -(grad_batch - grad_batch//4*3)] = 1
            sample_inds_aft[:, -(grad_batch - grad_batch//4*3):] = 1
            ### sample data part
            feat_w_grad_bfr, _ = self.encode_image(image[sample_inds_bfr.bool().reshape(-1)], b)
            with torch.no_grad():
                feat_wo_grad_mid, _ = self.encode_image(image[sample_inds_mid.bool().reshape(-1)], b)
            feat_w_grad_aft, _ = self.encode_image(image[sample_inds_aft.bool().reshape(-1)], b)
            cls_features = torch.cat([feat_w_grad_bfr, feat_wo_grad_mid, feat_w_grad_aft], dim=0)
            cls_features = cls_features.view(b, t, -1)
        else:
            cls_features, img_features = self.encode_image(image, b)
            cls_features = cls_features.view(b, t, -1)
        video_features = self.mit(cls_features, prompt, patch_info)
        return video_features, None

    def cache_text(self, text):
        self.eval()
        with torch.no_grad():
            if self.cache_text_features is None:
                self.cache_text_features = self.encode_text(text)
        self.train()
        return self.cache_text_features

    def cache_prompt(self, prompt):
        self.eval()
        with torch.no_grad():
            if self.cache_prompt_features is None:
                self.cache_prompt_features = self.encode_text(prompt)
        self.train()
        return self.cache_prompt_features

    ###
    # 101 start
    # 102 end
    # 132 ;
    # 3655 Unknown
    def aug_question(self, text, prompt, sample_cnt=0, TOKEN_NUM=None, is_training=False):
        TOKEN_NUM_STA = 101 if TOKEN_NUM is None else TOKEN_NUM['STA']
        TOKEN_NUM_END = 102 if TOKEN_NUM is None else TOKEN_NUM['END']
        TOKEN_NUM_SEP = 132 if TOKEN_NUM is None else TOKEN_NUM['SEP']
        TOKEN_NUM_UNK = 3655 if TOKEN_NUM is None else TOKEN_NUM['UNK']
        ###
        inds_delete = torch.LongTensor(random.sample(set(torch.arange(len(prompt)).tolist()), sample_cnt))
        inds_keep = torch.ones(len(prompt), device=prompt.device, dtype=torch.long)
        inds_keep[inds_delete] = 0
        prompt_new = prompt[inds_keep.bool()]
        splits_list = []
        for ind, iitem in enumerate(text):
            item = iitem[1: iitem.argmin(dim=-1) - 1]
            inds_split = torch.cat([torch.LongTensor([-1]).to(item.device),
                                    (item == TOKEN_NUM_SEP).int().nonzero().squeeze(dim=-1),
                                    torch.LongTensor([len(item)]).to(item.device)]).long()  # may without 0
            splits = []
            for i in range(len(inds_split) - 1):
                if i not in inds_delete and (item[inds_split[i]+1] != TOKEN_NUM_UNK):
                        splits.append(torch.cat([torch.LongTensor([TOKEN_NUM_SEP]).to(item.device),
                                                 item[inds_split[i]+1: inds_split[i+1]]]))
            if len(splits) > 0:
                splits = torch.cat(splits)[1:].long()
            else:
                splits = torch.LongTensor([]).to(item.device)
            splits_list.append(splits)
        if is_training:
            splits_text_list = [";".join([str(iispt.item()) for iispt in ispt]) for ispt in splits_list]
            splits_text_set = list(set(splits_text_list))
            label_hash_o2n = {_: splits_text_set.index(isp) for _, isp in enumerate(splits_text_list)}
            ### hash new text
            text_new = torch.zeros(len(splits_text_set), text.shape[1]).to(text.device, dtype=text.dtype)
            for ind, splits_text in enumerate(splits_text_set):
                if len(splits_text) == 0:
                    splits = torch.LongTensor([])
                else:
                    splits_text_rand = splits_text.split(";{};".format(TOKEN_NUM_SEP))
                    random.shuffle(splits_text_rand)
                    splits = torch.LongTensor([int(iispt) for iispt in (";{};".format(TOKEN_NUM_SEP)).join(splits_text_rand).split(";")])
                splits = torch.cat([torch.LongTensor([TOKEN_NUM_STA]).to(text.device),
                                    splits.to(text.device),
                                    torch.LongTensor([TOKEN_NUM_END]).to(text.device)]).long()
                text_new[ind, :len(splits)] = splits
            return text_new, prompt_new, label_hash_o2n
        else:
            text_new = torch.zeros_like(text)
            for ind, splits_long in enumerate(splits_list):
                splits = torch.cat([torch.LongTensor([TOKEN_NUM_STA]).to(text.device),
                                    splits_long.to(text.device),
                                    torch.LongTensor([TOKEN_NUM_END]).to(text.device)]).long()
                text_new[ind, :len(splits)] = splits
            return text_new, prompt_new, None

    # encode part text to reduce computational cost
    def encode_text_part(self, text, sample_cnt=96):
        if len(text) <= sample_cnt:
            return self.transformer(text, text != 0)
        inds_sample = random.sample(torch.arange(len(text)).tolist(), sample_cnt)
        inds_keep = torch.zeros(len(text)).bool()
        inds_keep[inds_sample] = True
        text_features_w_grad = self.transformer(text[inds_keep], text[inds_keep] != 0)
        with torch.no_grad():
            text_features_wo_grad = self.transformer(text[~inds_keep], text[~inds_keep] != 0)
        text_features = torch.zeros(len(text), text_features_w_grad.shape[1], device=text.device, dtype=text_features_w_grad.dtype)
        text_features[inds_keep] = text_features_w_grad
        text_features[~inds_keep] = text_features_wo_grad
        return text_features

    def encode_prompt_embed(self, prompt_embed):
        cur_model = self.transformer.model.model
        # embed
        input_shape = prompt_embed.size()
        batch_size, seq_length, _ = input_shape
        position_ids = cur_model.embeddings.position_ids[:, 0: seq_length]
        if hasattr(cur_model.embeddings, "token_type_ids"):
            buffered_token_type_ids = cur_model.embeddings.token_type_ids[:, :seq_length]
            buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
            token_type_ids = buffered_token_type_ids_expanded
        else:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=cur_model.embeddings.position_ids.device)

        inputs_embeds = prompt_embed
        token_type_embeddings = cur_model.embeddings.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if cur_model.embeddings.position_embedding_type == "absolute":
            position_embeddings = cur_model.embeddings.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = cur_model.embeddings.LayerNorm(embeddings)
        embedding_output = cur_model.embeddings.dropout(embeddings)

        # encode
        attention_mask = torch.ones(((batch_size, seq_length)), device=prompt_embed.device)
        extended_attention_mask = cur_model.get_extended_attention_mask(attention_mask, input_shape)
        head_mask = cur_model.get_head_mask(None, cur_model.config.num_hidden_layers)
        encoder_outputs = cur_model.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=True,
            return_dict=True,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = cur_model.pooler(sequence_output) if cur_model.pooler is not None else None
        bert_outputs = BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )
        # rest
        last_hidden_states = torch.stack([bert_outputs['hidden_states'][1], bert_outputs['hidden_states'][2],
                                          bert_outputs['hidden_states'][-1]])  # n_layer, batch, seqlen, emb_dim
        embed = last_hidden_states.permute(1, 0, 2, 3).mean(2).mean(1)  # pooling
        embed = self.transformer.projection_head(embed)
        return embed

    def forward(self, image, text, prompt, patch_info, imgs_embed):
        if patch_info['training']:
            text, prompt, label_hash_o2n = self.aug_question(text, prompt, random.randint(0, len(prompt)-1),
                                                             patch_info['TOKEN_NUM'], patch_info['training'])
            text_features = self.encode_text_part(text)
        else:
            text_features = self.transformer(text, text != 0)
        prompt_features = self.transformer(prompt, prompt != 0)
        ### learnable prompt
        # prompt_addition = self.prompt_learn_param
        prompt_addition = self.encode_prompt_embed(self.prompt_learn_param)
        prompt_addition = torch.cat([prompt_features, prompt_addition], dim=0)
        prompt_addition = prompt_addition / prompt_addition.norm(dim=-1, keepdim=True)

        b = image.shape[0]
        if self.is_img_pth:
            video_features = self.mit(imgs_embed, prompt_addition, patch_info)
        else:
            video_features, _ = self.encode_video(image, prompt_addition, patch_info)
            
        if patch_info['training']==True:
            label_id = patch_info['label_id']
            criterion = patch_info['criterion']
            video_features = video_features / video_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            ### ori cls
            text_features = text_features.unsqueeze(0).expand(b, -1, -1)
            logits = torch.einsum("bd,bkd->bk", video_features, text_features)
            # logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
            logit_scale = 300
            logits = logits * logit_scale
            label_id = torch.LongTensor([label_hash_o2n[_.item()] for _ in label_id]).to(label_id.device)
            loss = criterion(logits, label_id)
            return loss
        else:
            video_features = video_features / video_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.unsqueeze(0).expand(b, -1, -1)
            logits = torch.einsum("bd,bkd->bk", video_features, text_features)
            # logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
            logit_scale = 300
            logits = logits * logit_scale
            return logits

def build_model(state_dict: dict, T=8, droppath=0., use_checkpoint=False, logger=None, prompts_alpha=1e-1,
                prompts_layers=2, use_cache=True, mit_layers=1,
                context_length=77, T_mit=8, vit_vision_layers=-1, vit_text_layers=-1,
                is_img_pth=True,
                ):
    if not is_img_pth:
        ### L10 add for lora
        state_dict = OrderedDict(
            [(k.replace("visual.model", "visual").replace("transformer.model", "transformer"), v) for k, v in
             state_dict.items()])
        ###
        vit = "visual.proj" in state_dict
        if vit:
            vision_width = state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size
        else:
            counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
            vision_layers = tuple(counts)
            vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = None
            assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
            image_resolution = output_width * 32

        embed_dim = state_dict["text_projection"].shape[1]
        # L10 change context_length -> 4096
        if context_length != state_dict["positional_embedding"].shape[0]:
            pos_encodings = positional_encoding(context_length, state_dict["positional_embedding"].shape[1])
            pos_encodings = torch.Tensor(pos_encodings)
            state_dict["positional_embedding"] = pos_encodings
            # state_dict["positional_embedding"] = torch.randn(context_length, state_dict["positional_embedding"].shape[1])
        else:
            context_length = state_dict["positional_embedding"].shape[0]
        ###
        # context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

        if vit_vision_layers > 0:
            vision_layers = vit_vision_layers
        if vit_text_layers > 0:
            transformer_layers = vit_text_layers
    else:
        embed_dim = 512
        image_resolution = -1
        vision_layers = -1
        vision_width = -1
        vision_patch_size = -1
        vocab_size = -1
        transformer_width = -1
        transformer_heads = -1
        transformer_layers = -1
    model = XCLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,  
        T=T, droppath=droppath, mit_layers=mit_layers,
        prompts_alpha=prompts_alpha, prompts_layers=prompts_layers,
        use_checkpoint=use_checkpoint, use_cache=use_cache, T_mit=T_mit,
        is_img_pth=is_img_pth
    )

    # for key in ["input_resolution", "context_length", "vocab_size"]: # "mit.positional_embedding"
    #     if key in state_dict:
    #         del state_dict[key]

    # model.visual.model = LoraWrap(model.visual.model, "query,key,value,dense")
    # model.visual.projection_head.weight.requires_grad = True
    model.transformer.model = LoraWrap(model.transformer.model, "query,key,value,dense")
    model.transformer.projection_head.weight.requires_grad = True

    msg = model.load_state_dict(state_dict, strict=False)
    logger.info(f"load pretrained CLIP: {msg}")
    return model.eval()

def load(model_path, name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", 
         jit=True, T=8, droppath=0., use_checkpoint=False, logger=None, use_cache=True, prompts_alpha=1e-1,
         prompts_layers=2, mit_layers=1,
         context_length=77, T_mit=8, vit_vision_layers=-1, vit_text_layers=-1,
         is_img_pth = True,
):
    if model_path is None:
        model_path = clip._download(clip._MODELS[name])
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
        state_dict = None
    except RuntimeError:
        # loading saved state dict
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        # change by L10
        state_dict = torch.load(model_path, map_location="cpu")['model']
        # state_dict = torch.load(model_path, map_location="cpu")

    model = build_model(state_dict or model.state_dict(), T=T, droppath=droppath, 
                        use_checkpoint=use_checkpoint, logger=logger,
                        prompts_alpha=prompts_alpha, 
                        prompts_layers=prompts_layers,
                        use_cache=use_cache,
                        mit_layers=mit_layers,
                        context_length=context_length,
                        T_mit=T_mit,
                        vit_vision_layers=vit_vision_layers,
                        vit_text_layers=vit_text_layers,
                        is_img_pth=is_img_pth,
                        )
    if str(device) == "cpu":
        model.float()
    return model, model.state_dict()

def positional_encoding(max_len, d_model):
    pos_enc = np.zeros((max_len, d_model))
    for pos in range(max_len):
        for i in range(d_model):
            if i % 2 == 0:
                pos_enc[pos, i] = np.sin(pos / (10000 ** (2 * i / d_model)))
            else:
                pos_enc[pos, i] = np.cos(pos / (10000 ** ((2 * i - 1) / d_model)))
    return pos_enc