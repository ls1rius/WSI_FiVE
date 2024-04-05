import copy
import torch.optim as optim
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
import torch.distributed as dist

def is_main_process():
    return dist.get_rank() == 0

def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin

# def set_weight_decay(model, skip_list=(), skip_keywords=()):
#     has_decay = []
#     no_decay = []
# 
#     for name, param in model.named_parameters():
#         if not param.requires_grad:
#             continue  # frozen weights
#         if len(param.shape) == 1 or name.endswith('.bias') or (name in skip_list) or \
#                 check_keywords_in_name(name, skip_keywords):
#             no_decay.append(param)
#         else:
#             has_decay.append(param)
#     return [{'params': has_decay}, {'params': no_decay, 'weight_decay': 0.}]

def set_weight_decay(model, skip_list=(), skip_keywords=(), weight_decay=0.001, lr=2e-6, have=(), not_have=()):
    has_decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(have) > 0 and not check_keywords_in_name(name, have):
            continue
        if len(not_have) > 0 and check_keywords_in_name(name, not_have):
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
        else:
            has_decay.append(param)

    return [{'params': has_decay, 'weight_decay': weight_decay, 'lr': lr},
            {'params': no_decay, 'weight_decay': 0., 'lr': lr}]


# def fix_text(model):
#     for name, param in model.named_parameters():
#         if "visual." in name or "mit" in name or "prompts" in name:
#             continue
#         else:
#             param.requires_grad = False


# "positional_embedding" == name or \
def fix_text(model):
    for name, param in model.named_parameters():
        if name.startswith('transformer') or \
                "token_embedding.weight" == name or \
                "ln_final.weight" == name or \
                "ln_final.bias" == name or \
                "token_embedding" == name or \
                "text_projection" == name:
            param.requires_grad = False

def fix_image(model):
    for name, param in model.named_parameters():
        if "visual." in name:
            param.requires_grad = False

def unfix_lora(model):
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True

def fix_all(model):
    for name, param in model.named_parameters():
        param.requires_grad = False

from models.lora_wrap import mark_only_lora_as_trainable

# def build_optimizer(config, model):
#     model = model.module if hasattr(model, 'module') else model
#     parameters = set_weight_decay(model, {}, {})
#     optimizer = optim.AdamW(
#         parameters,
#         eps=1e-8,
#         betas=(0.9, 0.98),
#         lr=config.TRAIN.LR,
#         weight_decay=config.TRAIN.WEIGHT_DECAY)
# 
#     return optimizer

def build_optimizer(config, model):
    model = model.module if hasattr(model, 'module') else model

    # # fix text
    # if config.MODEL.FIX_TEXT:
    #     fix_text(model)
    #
    # # fix image
    # if config.MODEL.FIX_IMAGE:
    #     fix_image(model)

    unfix_lora(model)
    # model.positional_embedding.requires_grad = True

    # fix_all(model)
    # model.mit.project_for_cls.weight.requires_grad = True
    # model.mit.project_for_cls.bias.requires_grad = True

    # set decay and lr
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    clip_parameters = set_weight_decay(model, skip, skip_keywords,
        weight_decay=config.TRAIN.WEIGHT_DECAY, lr=config.TRAIN.LR,
        have=(), not_have=("prompts", "mit", "message_")
    )
    msg_parameters = set_weight_decay(model, skip, skip_keywords,
        weight_decay=config.TRAIN.WEIGHT_DECAY, lr=config.TRAIN.LR*10,
        have=("message_",), not_have=()
    )
    mit_parameters = set_weight_decay(model, skip, skip_keywords,
        weight_decay=config.TRAIN.WEIGHT_DECAY, lr=config.TRAIN.LR*10,
        have=("mit",), not_have=()
    )
    prompts_parameters = set_weight_decay(model, skip, skip_keywords,
        weight_decay=config.TRAIN.WEIGHT_DECAY, lr=config.TRAIN.LR*10,
        have=("prompts",), not_have=()
    )

    optimizer = optim.AdamW(clip_parameters + mit_parameters + prompts_parameters + msg_parameters,
                        betas=(0.9, 0.98), eps=1e-8,)
    # optimizer = optim.AdamW(mit_parameters,
    #                     betas=(0.9, 0.98), eps=1e-8,)

    return optimizer


def build_scheduler(config, optimizer, n_iter_per_epoch):
    num_steps = int(config.TRAIN.EPOCHS * n_iter_per_epoch)
    warmup_steps = int(config.TRAIN.WARMUP_EPOCHS * n_iter_per_epoch)

    # lr_scheduler = StepLRScheduler(
    #     optimizer=optimizer,
    #     decay_t=0.9,
    #     warmup_lr_init=0,
    #     warmup_t=warmup_steps,
    #     t_in_epochs=False,
    # )

    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=num_steps,
        lr_min=config.TRAIN.LR / 100,
        warmup_lr_init=0,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
    )

    # lr_scheduler = CosineLRScheduler(
    #     optimizer,
    #     t_initial=num_steps,
    #     lr_min=4e-5,
    #     warmup_lr_init=4e-6,
    #     warmup_t=warmup_steps,
    #     cycle_limit=1,
    #     t_in_epochs=False,
    # )

    return lr_scheduler