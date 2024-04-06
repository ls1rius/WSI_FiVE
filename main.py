import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import argparse
import datetime
import shutil
from pathlib import Path
from utils.config import get_config
from utils.optimizer import build_optimizer, build_scheduler
from utils.tools import AverageMeter, reduce_tensor, epoch_saving, load_checkpoint, generate_text, auto_resume_helper
from datasets.build import build_dataloader
from utils.logger import create_logger
import time
import numpy as np
import random
from apex import amp
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from datasets.blending import CutmixMixupBlending
from utils.config import get_config
from models import xclip
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.lora_wrap import LoraWrap
from utils.tools import five_scores

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', required=True, type=str, default='configs/k400/32_8.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--output', type=str, default="exp")
    parser.add_argument('--resume', type=str)
    parser.add_argument('--pretrained', type=str)
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--accumulation-steps', type=int)

    parser.add_argument("--local_rank", type=int, default=-1, help='local rank for DistributedDataParallel')
    args = parser.parse_args()

    config = get_config(args)
    ### add by L10
    config.defrost()
    config.DATA.NUM_CLASSES = len(pd.read_csv(config.DATA.LABEL_LIST))
    config.DATA.NUM_CLASSES_TRAIN = len(pd.read_csv(config.DATA.LABEL_LIST_TRAIN))
    config.DATA.NUM_CLASSES_VAL = len(pd.read_csv(config.DATA.LABEL_LIST_VAL))
    config.freeze()
    ###
    return args, config

def main(config):
    train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)
    os.environ["TOKENIZERS_PARALLELISM"] = "True"
    # tokenizer = AutoTokenizer.from_pretrained("./llama_hf/")
    model, _ = xclip.load(config.MODEL.PRETRAINED, config.MODEL.ARCH,
                            device="cpu", jit=False,
                            T=config.DATA.NUM_FRAMES,
                            droppath=config.MODEL.DROP_PATH_RATE,
                            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                            use_cache=config.MODEL.FIX_TEXT,
                            logger=logger,
                            mit_layers=1, # L10
                            context_length=config.MODEL.CONTEXT_LENGTH, # L10 add
                            T_mit=config.MODEL.GRAD_BATCH, # L10 change config.DATA.NUM_FRAMES -> config.MODEL.GRAD_BATCH
                            vit_vision_layers=config.MODEL.VIT_VISION_LAYERS, # L10 add
                            vit_text_layers=config.MODEL.VIT_TEXT_LAYERS,  # L10 add
                            is_img_pth=config.IS_IMG_PTH, # L10 add
                          )
    model = model.cuda()
    mixup_fn = None
    if config.AUG.MIXUP > 0:
        criterion = SoftTargetCrossEntropy()
        mixup_fn = CutmixMixupBlending(num_classes=config.DATA.NUM_CLASSES_TRAIN,
                                       smoothing=config.AUG.LABEL_SMOOTH, 
                                       mixup_alpha=config.AUG.MIXUP, 
                                       cutmix_alpha=config.AUG.CUTMIX, 
                                       switch_prob=config.AUG.MIXUP_SWITCH_PROB)
    elif config.AUG.LABEL_SMOOTH > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.AUG.LABEL_SMOOTH)
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(train_loader))
    if config.TRAIN.OPT_LEVEL != 'O0':
        model, optimizer = amp.initialize(models=model, optimizers=optimizer, opt_level=config.TRAIN.OPT_LEVEL)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False,
                                                      find_unused_parameters=True) # L10 change False->True

    start_epoch, max_accuracy = 0, 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        start_epoch, max_accuracy = load_checkpoint(config, model.module, optimizer, lr_scheduler, logger)

    text_labels_train = generate_text([item[0] for item in train_data.classes], config.MODEL.CONTEXT_LENGTH,
                                      tokenizer=model.module.transformer.tokenizer)
    text_labels_val = generate_text([item[0] for item in val_data.classes], config.MODEL.CONTEXT_LENGTH,
                                    tokenizer=model.module.transformer.tokenizer)
    prompts_labels = generate_text(config.PROMPT_LIST, config.MODEL.CONTEXT_LENGTH,
                                   tokenizer=model.module.transformer.tokenizer)
    # text_labels_train = generate_text(train_data.classes, config.MODEL.CONTEXT_LENGTH)
    # text_labels_val = generate_text(val_data.classes, config.MODEL.CONTEXT_LENGTH)
    # prompts_labels = generate_text(config.PROMPT_LIST, config.MODEL.CONTEXT_LENGTH)

    if config.TEST.ONLY_TEST:
        acc1 = validate(val_loader, text_labels_val, model,
                        prompts_labels,
                        config, config.DATA.NUM_CLASSES_VAL)
        logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {acc1:.1f}%")
        return

    for epoch in range(start_epoch, config.TRAIN.EPOCHS):
        train_loader.sampler.set_epoch(epoch)
        model.module.cache_text_features = None # add by L10
        model.module.cache_prompt_features = None # add by L10
        train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels_train,
                        prompts_labels,
                        config, mixup_fn)
        model.module.cache_text_features = None # add by L10
        model.module.cache_prompt_features = None # add by L10
        acc1 = validate(val_loader, text_labels_val, model,
                        prompts_labels,
                        config, config.DATA.NUM_CLASSES_VAL)
        logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {acc1:.1f}%")
        is_best = acc1 > max_accuracy
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            epoch_saving(config, epoch, model.module, max_accuracy, optimizer, lr_scheduler, logger, config.OUTPUT, is_best)

    train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)
    acc1 = validate(val_loader, text_labels_val, model,
                    prompts_labels,
                    config, config.DATA.NUM_CLASSES_VAL)
    logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {acc1:.1f}%")

def train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels,
                    prompts_features,
                    config, mixup_fn):
    model.train()
    optimizer.zero_grad()
    
    num_steps = len(train_loader)
    batch_time = AverageMeter()
    tot_loss_meter = AverageMeter()
    
    start = time.time()
    end = time.time()
    
    texts = text_labels.cuda(non_blocking=True)
    
    for idx, batch_data in enumerate(train_loader):
        images = batch_data["imgs"].cuda(non_blocking=True)
        label_id = batch_data["label"].cuda(non_blocking=True)
        sample_range = batch_data["sample_range"].cuda(non_blocking=True)
        posis = batch_data["posis"].cuda(non_blocking=True)
        patch_maxr = batch_data["patch_maxr"].cuda(non_blocking=True)
        patch_maxc = batch_data["patch_maxc"].cuda(non_blocking=True)
        imgs_embed = batch_data["imgs_embed"].cuda(non_blocking=True)
        patch_pub_cnt = batch_data["patch_pub_cnt"].cuda(non_blocking=True)
        patch_inds = batch_data["patch_inds"].cuda(non_blocking=True)

        label_id = label_id.reshape(-1)
        # images = images.view((-1, config.DATA.NUM_FRAMES, 3) + images.size()[-2:])

        patch_info = {
            'sample_range': sample_range,
            'posis': posis,
            'patch_maxr': patch_maxr,
            'patch_maxc': patch_maxc,
            'patch_pub_cnt': patch_pub_cnt,
            'patch_inds': patch_inds,
            'GRAD_BATCH': config.MODEL.GRAD_BATCH,
            'NUM_FRAMES': config.DATA.NUM_FRAMES,
            'training': True,
            'label_id': label_id,
            'criterion': criterion,
            'TOKEN_NUM': dict(config.TOKEN_NUM)
        }

        if mixup_fn is not None:
            images, label_id = mixup_fn(images, label_id)

        if texts.shape[0] == 1:
            texts = texts.view(1, -1)

        total_loss = model(images, texts, prompts_features, patch_info, imgs_embed)
        # output = model(images, texts, patch_info)
        # total_loss = criterion(output, label_id)
        total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS

        if config.TRAIN.ACCUMULATION_STEPS == 1:
            optimizer.zero_grad()
        if config.TRAIN.OPT_LEVEL != 'O0':
            with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            total_loss.backward()
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()
        
        tot_loss_meter.update(total_loss.item(), len(label_id))
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.9f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'tot_loss {tot_loss_meter.val:.4f} ({tot_loss_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

@torch.no_grad()
def validate(val_loader, text_labels, model,
             prompts_labels,
             config, num_classes):
    model.eval()
    
    acc1_meter, acc5_meter = AverageMeter(), AverageMeter()
    auc_meter, f1s_meter = AverageMeter(), AverageMeter()
    pred_label_list, pred_logist_list = [], []
    with torch.no_grad():
        text_inputs = text_labels.cuda()
        logger.info(f"{config.TEST.NUM_CLIP * config.TEST.NUM_CROP} views inference")
        for idx, batch_data in enumerate(val_loader):
            _image = batch_data["imgs"]
            label_id = batch_data["label"]
            sample_range = batch_data["sample_range"].cuda(non_blocking=True)
            posis = batch_data["posis"].cuda(non_blocking=True)
            patch_maxr = batch_data["patch_maxr"].cuda(non_blocking=True)
            patch_maxc = batch_data["patch_maxc"].cuda(non_blocking=True)
            imgs_embed = batch_data["imgs_embed"].cuda(non_blocking=True)
            patch_pub_cnt = batch_data["patch_pub_cnt"].cuda(non_blocking=True)
            patch_inds = batch_data["patch_inds"].cuda(non_blocking=True)

            label_id = label_id.reshape(-1)

            patch_info = {
                'sample_range': sample_range,
                'posis': posis,
                'patch_maxr': patch_maxr,
                'patch_maxc': patch_maxc,
                'patch_pub_cnt': patch_pub_cnt,
                'patch_inds': patch_inds,
                'GRAD_BATCH': config.MODEL.GRAD_BATCH,
                'NUM_FRAMES': config.DATA.NUM_FRAMES,
                'training': False,
                'label_id': None,
            }

            b, tn, c, h, w = _image.size()
            n = config.TEST.NUM_CLIP * config.TEST.NUM_CROP  # n = tn // t
            t = tn // n # t = config.DATA.NUM_FRAMES
            _image = _image.view(b, n, t, c, h, w)
           
            tot_similarity = torch.zeros((b, num_classes)).cuda()
            for i in range(n):
                image = _image[:, i, :, :, :, :] # [b,t,c,h,w]
                label_id = label_id.cuda(non_blocking=True)
                image_input = image.cuda(non_blocking=True)

                if config.TRAIN.OPT_LEVEL == 'O2':
                    image_input = image_input.half()
                
                output = model(image_input, text_inputs, prompts_labels, patch_info, imgs_embed)
                
                similarity = output.view(b, -1).softmax(dim=-1)
                tot_similarity += similarity

            values_1, indices_1 = tot_similarity.topk(1, dim=-1)
            values_5, indices_5 = tot_similarity.topk(min(5, num_classes), dim=-1)
            acc1, acc5 = 0, 0
            for i in range(b):
                if indices_1[i] == label_id[i]:
                    acc1 += 1
                if label_id[i] in indices_5[i]:
                    acc5 += 1
            acc1_meter.update(float(acc1) / b * 100, b)
            acc5_meter.update(float(acc5) / b * 100, b)
            if num_classes == 2:
                pred_label_list.extend(label_id.data.cpu().numpy())
                pred_logist_list.extend(tot_similarity[:, 1].data.cpu().numpy())
            if (idx % config.PRINT_FREQ == 0 and idx != 0) or (idx == len(val_loader) - 1):
                logger.info(
                    f'Test: [{idx}/{len(val_loader)}]\t'
                    f'Acc@1: {acc1_meter.avg:.3f}\t'
                    f'Acc@5: {acc5_meter.avg:.3f}\t'
                )
                if num_classes == 2:
                    acc_val, auc_val, pre_val, rec_val, f1s_val = five_scores(np.array(pred_label_list),
                                                                              np.array(pred_logist_list))
                    auc_meter.update(auc_val * 100)
                    f1s_meter.update(f1s_val * 100)
                    logger.info(
                        f'Test: [{idx}/{len(val_loader)}]\t'
                        f'Auc@1: {auc_meter.avg:.3f}\t'
                        f'F1S@1: {f1s_meter.avg:.3f}\t'
                    )
    acc1_meter.sync()
    acc5_meter.sync()
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    if num_classes == 2:
        auc_meter.sync()
        f1s_meter.sync()
        logger.info(f' * Auc@1 {auc_meter.avg:.3f} F1S@1 {f1s_meter.avg:.3f}')
    return acc1_meter.avg

if __name__ == '__main__':
    # prepare config
    args, config = parse_option()

    # init_distributed
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier(device_ids=[args.local_rank])

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # create working_dir
    Path(config.OUTPUT).mkdir(parents=True, exist_ok=True)
    
    # logger
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.ARCH}")
    logger.info(f"working dir: {config.OUTPUT}")
    
    # save config 
    if dist.get_rank() == 0:
        logger.info(config)
        shutil.copy(args.config, config.OUTPUT)

    main(config)