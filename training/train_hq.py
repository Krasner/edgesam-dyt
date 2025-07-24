# Copyright by HQ-SAM team
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import numpy as np
import datetime
from collections import defaultdict
import copy
import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import cv2
import random
from typing import Dict, List, Tuple

from edge_sam import build_sam_from_config, get_config, sam_model_registry
from optimizer import build_optimizer
from lr_scheduler import build_scheduler
from logger import create_logger
from utils import (
    add_common_args, load_checkpoint, load_pretrained, save_checkpoint,
    is_main_process, get_git_info,
    NativeScalerWithGradNormCount,
    sigmoid_ce_loss,
    dice_loss,
)
from my_meter import AverageMeter

from hq_utils.dataloader import get_im_gt_name_dict, create_dataloaders, RandomHFlip, Resize, LargeScaleJitter
from hq_utils.loss_mask import loss_masks
import hq_utils.misc as misc

def show_anns(masks, input_point, input_box, input_label, filename, image, ious, boundary_ious):
    if len(masks) == 0:
        return

    for i, (mask, iou, biou) in enumerate(zip(masks, ious, boundary_ious)):
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        if input_box is not None:
            show_box(input_box, plt.gca())
        if (input_point is not None) and (input_label is not None): 
            show_points(input_point, input_label, plt.gca())

        plt.axis('off')
        plt.savefig(filename+'_'+str(i)+'.png',bbox_inches='tight',pad_inches=-0.1)
        plt.close()

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


def parse_option():
    parser = argparse.ArgumentParser(
        'EdgeSAM training script', add_help=False)
    add_common_args(parser)
    args = parser.parse_args()

    config = get_config(args)

    return args, config

def is_valid_grad_norm(num):
    if num is None:
        return False
    return not bool(torch.isinf(num)) and not bool(torch.isnan(num))

def set_bn_state(config, model):
    if config.TRAIN.EVAL_BN_WHEN_TRAINING:
        for m in model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()

# def main(net, train_datasets, valid_datasets, args):
def main(args, config, train_datasets, valid_datasets):

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_sam_from_config(config, None, False, False)
    
    teacher_sam_checkpoint = "/home/ubuntu/edgesam-dyt/weights/sam_hq_vit_h.pth"
    teacher_model_type = "vit_h_hq"
    teacher_model = sam_model_registry[teacher_model_type](checkpoint=teacher_sam_checkpoint)
    teacher_model.eval()
    # teacher_model.compile()
    # teacher_model = torch.compile(teacher_model, mode="reduce-overhead")

    if not args.only_cpu:
        model.cuda()
        teacher_model.cuda()
    if args.use_sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # teacher_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    logger.info(str(model))

    optimizer = build_optimizer(config, model)

    loss_scaler = NativeScalerWithGradNormCount(grad_scaler_enabled=config.AMP_ENABLE)

    """
    misc.init_distributed_mode(args)
    print('world size: {}'.format(args.world_size))
    print('rank: {}'.format(args.rank))
    print('local_rank: {}'.format(args.local_rank))
    print("args: " + str(args) + '\n')

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    """
    ### --- Step 1: Train or Valid dataset ---
    if not args.eval:
        print("--- create training dataloader ---")
        train_im_gt_list = get_im_gt_name_dict(train_datasets, flag="train")
        train_dataloaders, train_datasets = create_dataloaders(train_im_gt_list,
                                                        my_transforms = [
                                                                    # Resize([1024,1024]),
                                                                    RandomHFlip(),
                                                                    LargeScaleJitter(
                                                                        # aug_scale_min=0.5,
                                                                        # aug_scale_max=1.25
                                                                    ) # resizes to 1024
                                                                    ],
                                                        batch_size = config.DATA.BATCH_SIZE,
                                                        training = True)
        print(len(train_dataloaders), " train dataloaders created")

    print("--- create valid dataloader ---")
    valid_im_gt_list = get_im_gt_name_dict(valid_datasets, flag="valid")
    valid_dataloaders, valid_datasets = create_dataloaders(valid_im_gt_list,
                                                          my_transforms = [
                                                                        Resize([1024,1024])
                                                                    ],
                                                          batch_size=1,
                                                          training=False)
    print(len(valid_dataloaders), " valid dataloaders created")
    
    lr_scheduler = build_scheduler(config, optimizer, len(
        train_dataloaders) // config.TRAIN.ACCUMULATION_STEPS)

    max_accuracy = 0.0

    ### --- Step 2: DistributedDataParallel---
    # if torch.cuda.is_available():
    #     net.cuda()
    # net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
    # net_without_ddp = net.module
    if not args.only_cpu:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False,
            find_unused_parameters=config.TRAIN.FIND_UNUSED_PARAMETERS)
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    # load image encoder weights
    load_pretrained(config, model_without_ddp.image_encoder, logger)
    # load_pretrained(config, model_without_ddp.prompt_encoder, logger, stage="prompt_encoer")
    load_pretrained(config, model_without_ddp.mask_decoder, logger, stage="mask_decoder")
    
    prompt_encoder_weights = torch.load('weights/sam_vit_h_prompt_encoder.pth', map_location='cpu')
    new_state_dict = {}
    for k, v in prompt_encoder_weights.items():
        _k = k.replace("prompt_encoder.","")
        new_state_dict[_k] = v
    msg = model_without_ddp.prompt_encoder.load_state_dict(new_state_dict, strict=False)
    # msg = model_without_ddp.prompt_encoder.load_state_dict(prompt_encoder_weights, strict=False)
    logger.warning(msg)
    logger.info(f"=> loaded successfully 'prompt_encoder'")
    del prompt_encoder_weights
    
    ### --- Freeze all submodel expect for HQ part of mask decoder
    for param in model_without_ddp.image_encoder.parameters():
        param.requires_grad = False
    model_without_ddp.image_encoder.eval()

    for param in model_without_ddp.prompt_encoder.parameters():
        param.requires_grad = False
    model_without_ddp.prompt_encoder.eval()
    
    for param in model_without_ddp.mask_decoder.parameters():
        param.requires_grad = False
    # breakpoint()
    # MANUALLY SET HF_TOKEN from teacher model
    model_without_ddp.mask_decoder.hf_token.weight = teacher_model.mask_decoder.hf_token.weight
    
    # MANUALLY copy over other matching weights and biases
    '''
    for i, layer in enumerate(model_without_ddp.mask_decoder.embedding_encoder):
        if hasattr(layer,'weight'):
            layer.weight = teacher_model.mask_decoder.embedding_encoder[i].weight
        if hasattr(layer,'bias'):
            layer.bias = teacher_model.mask_decoder.embedding_encoder[i].bias

    for i, layer in enumerate(model_without_ddp.mask_decoder.embedding_maskfeature):
        if hasattr(layer,'weight'):
            layer.weight = teacher_model.mask_decoder.embedding_maskfeature[i].weight
        if hasattr(layer,'bias'):
            layer.bias = teacher_model.mask_decoder.embedding_maskfeature[i].bias
    
    for i, layer in enumerate(model_without_ddp.mask_decoder.hf_mlp.layers):
        if hasattr(layer,'weight'):
            layer.weight = teacher_model.mask_decoder.hf_mlp.layers[i].weight
        if hasattr(layer,'bias'):
            layer.bias = teacher_model.mask_decoder.hf_mlp.layers[i].bias
    '''
    # breakpoint()
    # hf_token, hf_mlp, compress_vit_feat, embedding_encoder, embedding_maskfeature
    for name, param in model_without_ddp.mask_decoder.named_parameters():
        if (
            'hf_token' in name
            or 'hf_mlp' in name
            or 'compress_vit_feat' in name
            or 'embedding_encoder' in name
            or 'embedding_maskfeature' in name
        ):
            param.requires_grad = True
            print(f"{name} set to trainable")

    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")

    if torch.sum(model_without_ddp.mask_decoder.hf_token.weight - teacher_model.mask_decoder.hf_token.weight) == 0:
        logger.info("HF Token matches teacher as expected")
    
    ### --- Step 3: Train or Evaluate ---
    # if not args.eval:
        # print("--- define optimizer ---")
        # optimizer = optim.Adam(net_without_ddp.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop_epoch)
    # lr_scheduler.last_epoch = args.start_epoch
    
    loss_writer = None
    if dist.get_rank() == 0:
        loss_writer = SummaryWriter(f'{config.OUTPUT}/{datetime.datetime.now().strftime("%Y-%m-%d/%H:%M:%S")}')

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        train_step(args, config, model, optimizer, train_dataloaders, valid_dataloaders, teacher_model, epoch, lr_scheduler, loss_scaler, loss_writer)

        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp,
                            max_accuracy, optimizer, lr_scheduler, loss_scaler, logger)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

    """
    else:
        sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
        _ = sam.to(device=args.device)
        sam = torch.nn.parallel.DistributedDataParallel(sam, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)

        if args.restore_model:
            print("restore model from:", args.restore_model)
            if torch.cuda.is_available():
                net_without_ddp.load_state_dict(torch.load(args.restore_model))
            else:
                net_without_ddp.load_state_dict(torch.load(args.restore_model,map_location="cpu"))
    
        evaluate(args, net, sam, valid_dataloaders, args.visualize)
    """

def train_step(args, config, model, optimizer, train_dataloaders, valid_dataloaders, teacher_model, epoch, lr_scheduler, loss_scaler, loss_writer):

    model.train()
    set_bn_state(config, model)
    optimizer.zero_grad()
    
    # sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    # _ = sam.to(device=args.device)
    # sam = torch.nn.parallel.DistributedDataParallel(sam, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()
    meters = defaultdict(AverageMeter)

    start = time.time()
    end = time.time()
    data_tic = time.time()

    num_steps = len(train_dataloaders)

    print("epoch:   ",epoch, "  learning rate:  ", optimizer.param_groups[0]["lr"])
    metric_logger = misc.MetricLogger(delimiter="  ")
    train_dataloaders.batch_sampler.sampler.set_epoch(epoch)

    # for idx, data in enumerate(metric_logger.log_every(train_dataloaders,10)):
    for idx, data in enumerate(train_dataloaders):
        
        meters['data_time'].update(time.time() - data_tic)

        inputs, labels = data['image'], data['label']
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
            
        imgs = inputs.permute(0, 2, 3, 1).cpu().numpy()
        batch_size = imgs.shape[0]
        
        # input prompt
        # we only train with box and point prompts right now...
        input_keys = ['box','point'] # ,'noise_mask']
        # try:
            # unlike sam-hq sample both positive and negative points
            # positive
        pos_labels_points = misc.masks_sample_points(labels[:,0,:,:], k=5) # (b, n, 2)
        # negative
        neg_labels_points = misc.masks_sample_points(labels[:,0,:,:], k=5, positive=False)
        
        labels_box = misc.masks_to_boxes(labels[:,0,:,:])
        # except:
            # less than 10 points
        #     input_keys = ['box'] # ,'noise_mask']
        labels_256 = F.interpolate(labels, size=(256, 256), mode='bilinear')
        labels_noisemask = misc.masks_noise(labels_256)

        batched_input = []
        keep_labels = []
        for b_i in range(len(imgs)):
            dict_input = dict()
            input_image = torch.as_tensor(imgs[b_i].astype(dtype=np.uint8), device=model.device).permute(2, 0, 1).contiguous().float()
            if torch.any(torch.isnan(input_image)):
                continue
            dict_input['image'] = input_image # .requires_grad_(True)
            input_type = random.choice(input_keys)

            # chance to include box with points
            # if ((input_type == 'box') or (input_type == 'point' and torch.rand(()) < 0.25)):
            
            _box = labels_box[b_i:b_i+1]
            if (_box[0,0] == 1e8) or (_box[0,1] == 1e8):
                _box = None

            # dict_input['boxes'] = _box # .requires_grad_(True)

            # elif input_type == 'point':
            pos_point_coords = pos_labels_points[b_i]# :b_i+1]
            neg_point_coords = neg_labels_points[b_i]# :b_i+1]
            # if torch.any(pos_point_coords == 1e8) or torch.any(neg_point_coords == 1e8):
            #     continue
            if torch.all(pos_point_coords == 1e8):
                point_coords, point_labels = None, None
            else:
                valid_pos = ~torch.any(pos_point_coords == 1e8, 1)
                valid_neg = ~torch.any(neg_point_coords == 1e8, 1)

                pos_point_coords = pos_point_coords[valid_pos].unsqueeze(0)
                neg_point_coords = neg_point_coords[valid_neg].unsqueeze(0)
                
                pos_labels = torch.ones(pos_point_coords.shape[1], device=pos_point_coords.device)[None,:]
                neg_labels = torch.zeros(neg_point_coords.shape[1], device=neg_point_coords.device)[None,:]
                
                # select at least 1 positive point
                # min(5, pos_point_coords.shape[1])
                n_pos = torch.randint(1, pos_point_coords.shape[1], ())
                if (neg_point_coords.shape[1] > 0) and (torch.rand(()) > 0.5):
                    n_neg = torch.randint(0, neg_point_coords.shape[1], ())
                else:
                    n_neg = 0

                point_coords = torch.cat((pos_point_coords[:, :n_pos], neg_point_coords[:, :n_neg]), 1)
                point_labels = torch.cat((pos_labels[:, :n_pos], neg_labels[:, :n_neg]), 1)
            # breakpoint()
            
            input_box = _box if torch.rand(()) > 0.5 else None
            
            if (input_box is not None) and torch.rand(()) < 0.25:
                point_coords, point_labels = None, None

            if (point_coords is None) and (input_box is None):
                input_box = _box

            # breakpoint()
            dict_input['boxes'] = input_box
            if point_coords is not None:
                dict_input['point_coords'] = point_coords # .requires_grad_(True)
                dict_input['point_labels'] = point_labels # torch.ones(point_coords.shape[1], device=point_coords.device)[None,:]# .requires_grad_(True)
                
            # elif input_type == 'noise_mask':
            #     dict_input['mask_inputs'] = labels_noisemask[b_i:b_i+1]# .requires_grad_(True)
            # else:
            #     raise NotImplementedError
            dict_input['original_size'] = imgs[b_i].shape[:2]
            batched_input.append(dict_input)
            keep_labels.append(labels[b_i:b_i+1])

        if len(keep_labels) == 0:
            print("All prompts were invalid... skipping")
            continue

        keep_labels = torch.cat(keep_labels, 0)
        # breakpoint()
        with torch.autograd.set_detect_anomaly(True):
            
            # some masks_hq are NaN???
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=config.AMP_ENABLE, cache_enabled=False):
                outs = model(batched_input, num_multimask_outputs=1, training_mode=True)
                # outs is list [batch_size] of dict with keys ['masks', 'iou_predictions', 'low_res_logits']
                masks_hq = torch.concat([o['low_res_logits'] for o in outs], 0)
                valid = [~torch.any(torch.isnan(m)) for m in masks_hq]
                # print(valid)
                masks_s = torch.stack([m for i, m in enumerate(masks_hq) if valid[i]], 0)
                
            _labels = torch.stack([l/255.0 for i, l in enumerate(keep_labels) if valid[i]], 0)
            # loss_mask, loss_dice = loss_masks(masks_hq, labels/255.0, len(masks_hq))
            loss_mask, loss_dice = loss_masks(masks_s.float(), _labels.float(), len(masks_s))
            
            total_loss = loss_mask + loss_dice

            if torch.isnan(total_loss):
                breakpoint()

            loss_dict = {
                "loss_mask": loss_mask, 
                "loss_dice":loss_dice,
            }

            distill_loss_mask = 0.0
            distill_loss_dice = 0.0
            if config.TRAIN.ENABLE_DISTILL:
                # distillation from teacher
                with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16, enabled=config.AMP_ENABLE):
                    teacher_outs = teacher_model(batched_input, num_multimask_outputs=1)
                    masks_t = torch.concat([o['low_res_logits'] for i, o in enumerate(teacher_outs) if valid[i]], 0)

                if config.DISTILL.DECODER_BCE > 0 or config.DISTILL.DECODER_FOCAL > 0 or config.DISTILL.DECODER_DICE > 0:
                    _mask_s = masks_s.float()
                    _mask_t = masks_t.float()

                    temperature = config.DISTILL.TEMPERATURE
                    _mask_s = _mask_s / temperature
                    _mask_t = _mask_t / temperature

                    target_logit = True
                    if not config.DISTILL.USE_TEACHER_LOGITS:
                        _mask_t = (_mask_t > 0.0).float()
                        target_logit = False
                    
                    if config.DISTILL.DECODER_BCE > 0:
                        distill_loss_mask = sigmoid_ce_loss(_mask_s, _mask_t, None, target_logit) * (temperature ** 2) * config.DISTILL.DECODER_BCE

                    if config.DISTILL.DECODER_DICE > 0:
                        distill_loss_dice = dice_loss(_mask_s, _mask_t, None, target_logit) * (temperature ** 2) * config.DISTILL.DECODER_DICE
                
                    # target_labels = (_mask_t > 0).float()
                    # distill_loss_mask, distill_loss_dice = loss_masks(_mask_s, target_labels, len(masks_hq))
                    # distill_loss_mask *= config.DISTILL.DECODER_BCE
                    # distill_loss_dice *= config.DISTILL.DECODER_DICE

                    total_loss = total_loss + distill_loss_mask + distill_loss_dice

                loss_dict.update({
                    "distill_loss_mask": distill_loss_mask,
                    "distill_loss_dice": distill_loss_dice,
                })

            # print(f"loss_mask: {loss_mask.item()}")
            # print(f"loss_dice: {loss_dice.item()}")
            # print(loss_dict)

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = misc.reduce_dict(loss_dict)
            losses_reduced_scaled = sum(loss_dict_reduced.values())
            loss_value = losses_reduced_scaled.item()
            meters['total_loss'].update(loss_value, len(masks_s))

            if loss_writer is not None:
                display_dict = {'total': total_loss}
                for key in loss_dict:
                    display_dict[key] = loss_dict[key].item()

                loss_writer.add_scalars('loss', display_dict, epoch * num_steps + idx)

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            
            grad_norm = loss_scaler(total_loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.zero_grad()
                lr_scheduler.step_update(
                    (epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
            loss_scale_value = loss_scaler.state_dict().get("scale", 1.0)

        torch.cuda.synchronize()

        if not torch.isnan(total_loss):
            loss_meter.update(total_loss.item(), batch_size)
        if is_valid_grad_norm(grad_norm):
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()
        data_tic = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)

            extra_meters_str = ''
            for k, v in meters.items():
                extra_meters_str += f'{k} {v.val:.4f} ({v.avg:.4f})\t'
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'{extra_meters_str}'
                f'mem {memory_used:.0f}MB')

    epoch_time = time.time() - start
    extra_meters_str = f'Train-Summary: [{epoch}/{config.TRAIN.EPOCHS}]\t'
    for k, v in meters.items():
        v.sync()
        extra_meters_str += f'{k} {v.val:.4f} ({v.avg:.4f})\t'
    logger.info(extra_meters_str)
    logger.info(
        f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

    # metric_logger.update(training_loss=loss_value, **loss_dict_reduced)
    # print("Finished epoch:      ", epoch)
    # metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    # train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    # lr_scheduler.step()
    # test_stats = evaluate(args, net, sam, valid_dataloaders)
    # train_stats.update(test_stats)
    """
    net.train()  

    if epoch % args.model_save_fre == 0:
        model_name = "/epoch_"+str(epoch)+".pth"
        print('come here save at', args.output + model_name)
        misc.save_on_master(net.module.state_dict(), args.output + model_name)

    # Finish training
    print("Training Reaches The Maximum Epoch Number")
    
    # merge sam and hq_decoder
    if is_main_process():
        sam_ckpt = torch.load(args.checkpoint)
        hq_decoder = torch.load(args.output + model_name)
        for key in hq_decoder.keys():
            sam_key = 'mask_decoder.'+key
            if sam_key not in sam_ckpt.keys():
                sam_ckpt[sam_key] = hq_decoder[key]
        model_name = "/sam_hq_epoch_"+str(epoch)+".pth"
        torch.save(sam_ckpt, args.output + model_name)
    """


def compute_iou(preds, target):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    for i in range(0,len(preds)):
        iou = iou + misc.mask_iou(postprocess_preds[i],target[i])
    return iou / len(preds)

def compute_boundary_iou(preds, target):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    for i in range(0,len(preds)):
        iou = iou + misc.boundary_iou(target[i],postprocess_preds[i])
    return iou / len(preds)

def evaluate(args, net, sam, valid_dataloaders, visualize=False):
    net.eval()
    print("Validating...")
    test_stats = {}

    for k in range(len(valid_dataloaders)):
        metric_logger = misc.MetricLogger(delimiter="  ")
        valid_dataloader = valid_dataloaders[k]
        print('valid_dataloader len:', len(valid_dataloader))

        for data_val in metric_logger.log_every(valid_dataloader,1000):
            imidx_val, inputs_val, labels_val, shapes_val, labels_ori = data_val['imidx'], data_val['image'], data_val['label'], data_val['shape'], data_val['ori_label']

            if torch.cuda.is_available():
                inputs_val = inputs_val.cuda()
                labels_val = labels_val.cuda()
                labels_ori = labels_ori.cuda()

            imgs = inputs_val.permute(0, 2, 3, 1).cpu().numpy()
            
            labels_box = misc.masks_to_boxes(labels_val[:,0,:,:])
            input_keys = ['box']
            batched_input = []
            for b_i in range(len(imgs)):
                dict_input = dict()
                input_image = torch.as_tensor(imgs[b_i].astype(dtype=np.uint8), device=sam.device).permute(2, 0, 1).contiguous()
                dict_input['image'] = input_image 
                input_type = random.choice(input_keys)
                if input_type == 'box':
                    dict_input['boxes'] = labels_box[b_i:b_i+1]
                elif input_type == 'point':
                    point_coords = labels_points[b_i:b_i+1]
                    dict_input['point_coords'] = point_coords
                    dict_input['point_labels'] = torch.ones(point_coords.shape[1], device=point_coords.device)[None,:]
                elif input_type == 'noise_mask':
                    dict_input['mask_inputs'] = labels_noisemask[b_i:b_i+1]
                else:
                    raise NotImplementedError
                dict_input['original_size'] = imgs[b_i].shape[:2]
                batched_input.append(dict_input)

            with torch.no_grad():
                batched_output, interm_embeddings = sam(batched_input, multimask_output=False)
            
            batch_len = len(batched_output)
            encoder_embedding = torch.cat([batched_output[i_l]['encoder_embedding'] for i_l in range(batch_len)], dim=0)
            image_pe = [batched_output[i_l]['image_pe'] for i_l in range(batch_len)]
            sparse_embeddings = [batched_output[i_l]['sparse_embeddings'] for i_l in range(batch_len)]
            dense_embeddings = [batched_output[i_l]['dense_embeddings'] for i_l in range(batch_len)]
            
            masks_sam, masks_hq = net(
                image_embeddings=encoder_embedding,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                hq_token_only=False,
                interm_embeddings=interm_embeddings,
            )

            iou = compute_iou(masks_hq,labels_ori)
            boundary_iou = compute_boundary_iou(masks_hq,labels_ori)

            if visualize:
                print("visualize")
                os.makedirs(args.output, exist_ok=True)
                masks_hq_vis = (F.interpolate(masks_hq.detach(), (1024, 1024), mode="bilinear", align_corners=False) > 0).cpu()
                for ii in range(len(imgs)):
                    base = data_val['imidx'][ii].item()
                    print('base:', base)
                    save_base = os.path.join(args.output, str(k)+'_'+ str(base))
                    imgs_ii = imgs[ii].astype(dtype=np.uint8)
                    show_iou = torch.tensor([iou.item()])
                    show_boundary_iou = torch.tensor([boundary_iou.item()])
                    show_anns(masks_hq_vis[ii], None, labels_box[ii].cpu(), None, save_base , imgs_ii, show_iou, show_boundary_iou)
                       

            loss_dict = {"val_iou_"+str(k): iou, "val_boundary_iou_"+str(k): boundary_iou}
            loss_dict_reduced = misc.reduce_dict(loss_dict)
            metric_logger.update(**loss_dict_reduced)


        print('============================')
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
        test_stats.update(resstat)


    return test_stats


if __name__ == "__main__":

    ### --------------- Configuring the Train and Valid datasets ---------------

    dataset_dis = {"name": "DIS5K-TR",
                 "im_dir": "./datasets/DIS5K/DIS-TR/im",
                 "gt_dir": "./datasets/DIS5K/DIS-TR/gt",
                 "im_ext": ".jpg",
                 "gt_ext": ".png"}

    dataset_thin = {"name": "ThinObject5k-TR",
                 "im_dir": "./datasets/thin_object_detection/ThinObject5K/images_train",
                 "gt_dir": "./datasets/thin_object_detection/ThinObject5K/masks_train",
                 "im_ext": ".jpg",
                 "gt_ext": ".png"}

    dataset_fss = {"name": "FSS",
                 "im_dir": "./datasets/cascade_psp/fss_all",
                 "gt_dir": "./datasets/cascade_psp/fss_all",
                 "im_ext": ".jpg",
                 "gt_ext": ".png"}

    dataset_duts = {"name": "DUTS-TR",
                 "im_dir": "./datasets/cascade_psp/DUTS-TR",
                 "gt_dir": "./datasets/cascade_psp/DUTS-TR",
                 "im_ext": ".jpg",
                 "gt_ext": ".png"}

    dataset_duts_te = {"name": "DUTS-TE",
                 "im_dir": "./datasets/cascade_psp/DUTS-TE",
                 "gt_dir": "./datasets/cascade_psp/DUTS-TE",
                 "im_ext": ".jpg",
                 "gt_ext": ".png"}

    dataset_ecssd = {"name": "ECSSD",
                 "im_dir": "./datasets/cascade_psp/ecssd",
                 "gt_dir": "./datasets/cascade_psp/ecssd",
                 "im_ext": ".jpg",
                 "gt_ext": ".png"}

    dataset_msra = {"name": "MSRA10K",
                 "im_dir": "./datasets/cascade_psp/MSRA_10K",
                 "gt_dir": "./datasets/cascade_psp/MSRA_10K",
                 "im_ext": ".jpg",
                 "gt_ext": ".png"}

    # valid set
    dataset_coift_val = {"name": "COIFT",
                 "im_dir": "./datasets/thin_object_detection/COIFT/images",
                 "gt_dir": "./datasets/thin_object_detection/COIFT/masks",
                 "im_ext": ".jpg",
                 "gt_ext": ".png"}

    dataset_hrsod_val = {"name": "HRSOD",
                 "im_dir": "./datasets/thin_object_detection/HRSOD/images",
                 "gt_dir": "./datasets/thin_object_detection/HRSOD/masks_max255",
                 "im_ext": ".jpg",
                 "gt_ext": ".png"}

    dataset_thin_val = {"name": "ThinObject5k-TE",
                 "im_dir": "./datasets/thin_object_detection/ThinObject5K/images_test",
                 "gt_dir": "./datasets/thin_object_detection/ThinObject5K/masks_test",
                 "im_ext": ".jpg",
                 "gt_ext": ".png"}

    dataset_dis_val = {"name": "DIS5K-VD",
                 "im_dir": "./datasets/DIS5K/DIS-VD/im",
                 "gt_dir": "./datasets/DIS5K/DIS-VD/gt",
                 "im_ext": ".jpg",
                 "gt_ext": ".png"}

    train_datasets = [dataset_dis, dataset_thin, dataset_fss, dataset_duts, dataset_duts_te, dataset_ecssd, dataset_msra]
    valid_datasets = [dataset_dis_val, dataset_coift_val, dataset_hrsod_val, dataset_thin_val] 

    args, config = parse_option()
    config.defrost()
    config.freeze()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    if args.only_cpu:
        ddp_backend = 'gloo'
    else:
        torch.cuda.set_device(config.LOCAL_RANK)
        ddp_backend = 'nccl'

    torch.distributed.init_process_group(
        backend=ddp_backend, init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    # HQ-SAM uses a default batch size of 4
    linear_scaled_lr = config.TRAIN.BASE_LR * \
                       config.DATA.BATCH_SIZE * dist.get_world_size() / 4.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * \
                              config.DATA.BATCH_SIZE * dist.get_world_size() / 4.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * \
                           config.DATA.BATCH_SIZE * dist.get_world_size() / 4.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT,
                           dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")
    
    if is_main_process():
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

        config_dict = dict(config)
        config_dict['git'] = get_git_info()
        
    # print git info
    logger.info('===== git =====')
    logger.info(str(get_git_info()))

    # print config
    logger.info(config.dump())

    main(args, config, train_datasets, valid_datasets)
