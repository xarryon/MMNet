import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
from pathlib import Path
import sys
import argparse
import torch
import numpy as np
import random
import logging
from types import MethodType
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from datasets.dataset_test import SeqDeepFakeDataset
from tools.utils import AverageMeter, NestedTensor

from tools.env import init_dist
import torch.multiprocessing as mp

from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.configuration import Config
from models import mmnet as MMNet

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tools.loss import con_loss_extra, con_loss_inner, focal_loss

def setlogger(log_file):
    filehandler = logging.FileHandler(log_file)
    streamhandler = logging.StreamHandler()

    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)

    def epochInfo(self, set, idx, acc_fixed, acc_adaptive):
        self.info('{set}-{idx:d} epoch | acc_fixed:{acc_fixed:.4f}% | acc_adaptive:{acc_adaptive:.4f}%'.format(
            set=set,
            idx=idx,
            acc_fixed=acc_fixed,
            acc_adaptive=acc_adaptive
        ))

    logger.epochInfo = MethodType(epochInfo, logger)

    return logger


def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok = True)


def preset_model(args, cfg, model, logger, sum_steps):
    param_dicts = [
        {"params": [p for n, p in model.named_parameters(
        ) if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": cfg.lr_backbone,
        },
    ]         

    optimizer = torch.optim.AdamW(
        param_dicts, lr=cfg.lr, weight_decay=cfg.weight_decay)
        
    if cfg.warmup:
        warm_up_with_multistep_lr = lambda epoch: (epoch+1) / cfg.warmup_epochs if epoch < cfg.warmup_epochs else 0.1**len([m for m in cfg.lr_milestones if m <= epoch])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_multistep_lr)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150,eta_min=1e-5)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.lr_drop)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        start_epoch = checkpoint['epoch']
        
        if args.log:
            logger.info(f'Loading model from {args.resume}...')
            logger.info(f'start_epoch: {start_epoch}...')

        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda(args.gpu)

        # scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        if args.log:
            logger.info('Create new model')

        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        start_epoch = 0

    return model, optimizer, start_epoch, scheduler
    # return model, optimizer, start_epoch


def read_csv(field, file):
    info = pd.read_csv(file)
    image_list = info[field[0]].tolist()
    score_list = info[field[1]].tolist()
    return image_list, score_list


def evalute(cfg, val_dataloader, model):
    # switch model to evaluation mode
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=cfg.PAD_token_id)
    criterion.eval()
    total = len(val_dataloader)

    with torch.no_grad():
        validation_loss = 0.0

        for steps, (images, masks, caps, cap_masks) in enumerate(tqdm(val_dataloader)):
            
            samples = NestedTensor(images, masks).to(0)
            caps = caps.cuda()
            cap_masks = cap_masks.cuda()

            input_caps = caps[:, :-1]
            pad_token_input_caps = cfg.PAD_token_id*torch.ones_like(input_caps)
            input_caps = torch.where(input_caps==cfg.EOS_token_id, pad_token_input_caps, input_caps)
            input_cap_masks = input_caps==cfg.PAD_token_id

            outputs = model(samples, input_caps, input_cap_masks)
            loss = criterion(outputs.permute(0, 2, 1), caps[:, 1:])


            validation_loss += loss.item()
            
        loss = validation_loss / total

    return loss


def create_caption_and_mask(cfg):
    caption_template = cfg.PAD_token_id*torch.ones((1, cfg.max_position_embeddings), dtype=torch.long).cuda()
    mask_template = torch.ones((1, cfg.max_position_embeddings), dtype=torch.bool).cuda()

    caption_template[:, 0] = cfg.SOS_token_id
    mask_template[:, 0] = False

    return caption_template, mask_template


def evalute_transformer(cfg, val_dataloader, model):
    # switch model to evaluation mode
    model.eval()

    with torch.no_grad():
        running_corrects_fixed = 0.0
        epoch_size_fixed = 0.0 

        running_corrects_adaptive = 0.0
        epoch_size_adaptive = 0.0 

        for steps, (image, labels) in enumerate(tqdm(val_dataloader)):
            caption, cap_mask = create_caption_and_mask(cfg)
            image, labels = image.cuda(), labels.long().cuda()
            
            for i in range(cfg.max_position_embeddings - 1):
                # predictions = model(image, caption, cap_mask)
                # predictions, _, _ = model(image, caption, cap_mask)
                predictions, _, _ = model(image, caption, cap_mask)
                # predictions = predictions[:, i, :]
                predictions1 = nn.Softmax(dim=1)(predictions[0][:, i, :])
                predictions2 = nn.Softmax(dim=1)(predictions[1][:, i, :])
                predictions3 = nn.Softmax(dim=1)(predictions[2][:, i, :])
                
                score1 = torch.max(predictions1, dim=-1)[0]
                score2 = torch.max(predictions2, dim=-1)[0]
                score3 = torch.max(predictions3, dim=-1)[0]
                
                idx = torch.argmax(torch.Tensor([score1, score2, score3]))
                predicted_id = torch.argmax(predictions[idx][:,i,:], axis=-1)

                if predicted_id[0] == cfg.EOS_token_id:
                    caption = caption[:, 1:]
                    zero = torch.zeros_like(caption)
                    caption = torch.where(caption==cfg.PAD_token_id, zero, caption)
                    break

                caption[:, i+1] = predicted_id[0]
                cap_mask[:, i+1] = False
            
            if caption.shape[1] == 6:
                caption = caption[:, 1:]

            running_corrects_fixed += torch.sum(caption.cpu() == labels.data.cpu())
            epoch_size_fixed += image.size(0)*labels.shape[1]

            cmp_len = max(len(torch.where(labels[0]>0)[0]), len(torch.where(caption[0]>0)[0]))
            if cmp_len == 0:
                running_corrects_adaptive += 1
                cmp_len = 1
            else:
                running_corrects_adaptive += torch.sum(caption[:,:cmp_len].cpu() == labels[:,:cmp_len].data.cpu())
            epoch_size_adaptive += image.size(0)*cmp_len

    ACC_fixed =  running_corrects_fixed.double() / epoch_size_fixed
    ACC_adaptive =  running_corrects_adaptive.double() / epoch_size_adaptive

    return ACC_fixed, ACC_adaptive

def set_lr(cfg, epoch, base, optimizer):
    min_lrs = 1e-5

    if epoch < cfg.warmup_epochs:
        tmp_lr = base * (epoch+1) / cfg.warmup_epochs
        t = 0
        for param_group in optimizer.param_groups:
            param_group['lr'] = tmp_lr if t == 0 else tmp_lr * 0.1
            t += 1
    else:
        if epoch >= cfg.warmup_epochs and epoch <= 170:
            tmp_lr = tmp_lr = min_lrs + 0.5*(base-min_lrs)*(1+math.cos(math.pi*(epoch-20)*1./(170-20)))
            t = 0
            for param_group in optimizer.param_groups:
                param_group['lr'] = tmp_lr if t == 0 else tmp_lr * 0.1
                t += 1

        elif epoch > 170:
            t = 0
            for param_group in optimizer.param_groups:
                param_group['lr'] = min_lrs if t == 0 else min_lrs * 0.1
                t += 1

def train_ce(args, cfg, model, train_dataloader, optimizer, criterion, loss_logger, summary_writer, global_step, logger):
    for steps, (images, masks, caps, cap_masks, length, seqlabel, gt_region) in enumerate(train_dataloader):
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        
        samples = NestedTensor(images, masks).to(args.gpu)
        caps = caps.cuda(args.gpu)
        cap_masks = cap_masks.cuda(args.gpu)
        length = length.cuda(args.gpu)
        seqlabel = seqlabel.cuda(args.gpu)
        gt_region = gt_region.cuda(args.gpu)
        
        input_caps = caps[:, :-1]
        pad_token_input_caps = cfg.PAD_token_id*torch.ones_like(input_caps)
        input_caps = torch.where(input_caps==cfg.EOS_token_id, pad_token_input_caps, input_caps)
        input_cap_masks = input_caps==cfg.PAD_token_id
        
        outputs, src_c, fake_region = model(samples, input_caps, input_cap_masks)
        
        loss1 = criterion[0](outputs[0].permute(0, 2, 1), caps[:, 1:])
        loss2 = criterion[1](outputs[1].permute(0, 2, 1), caps[:, 1:])
        loss3 = criterion[2](outputs[2].permute(0, 2, 1), caps[:, 1:])
        
        loss4 = criterion[3](fake_region, gt_region)
        
        loss = loss1 + loss2 + loss3 + loss4
        
        if not math.isfinite(loss):
            print(f'Loss is {loss}, stopping training')
            sys.exit(1)
                
        optimizer.zero_grad()
        loss.backward()
                
        if cfg.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_max_norm)

        optimizer.step()
        loss_logger.update(loss.item(), images.size(0))

        global_step+=1

        #============ tensorboard train log info ============#
        if args.log:
            lossinfo = {
                'Train_Loss': loss.item(),                                                                                                  
                'Train_Loss_avg': loss_logger.avg,                                                                                                  
                    } 

            #============ print the train log info ============# 
            train_dataloader.set_description(
                'lr: {lr:.8f} | loss: {loss:.8f} '.format(
                    loss=loss_logger.avg,
                    lr = current_lr
                )
            )

    if args.log:
        logger.info(f'## Seql(1|2|3):{loss1.item()}|{loss2.item()}|{loss3.item()} || Reg:{loss4.item()} || Loss: {loss.item()} ##')

def compute_score(pred, caption, cfg):
    
    zeros = torch.zeros_like(caption)
    minus = zeros - 1
    minus2 = zeros - 2
    caption = torch.where(caption==cfg.PAD_token_id, minus, caption) 
    caption = torch.where(caption==cfg.EOS_token_id, minus, caption)
    pred = torch.where(pred==cfg.EOS_token_id, minus2, pred)
    score = (torch.sum(pred == caption, dim=1)+1e-6)/(torch.maximum(torch.sum(pred>0, dim=-1), torch.sum(caption>0, dim=-1))+1e-6)
    
    return score

def train_scst(args, cfg, model, train_dataloader, optimizer, loss_logger, summary_writer, global_step):
    for steps, (images, masks, caps, cap_masks, length, seqlabel, gt_region) in enumerate(train_dataloader):
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        
        samples = NestedTensor(images, masks).to(args.gpu)
        caps = caps.cuda(args.gpu)
        cap_masks = cap_masks.cuda(args.gpu)

        input_caps = caps[:, :-1]
        pad_token_input_caps = cfg.PAD_token_id * torch.ones_like(input_caps)
        input_caps = torch.where(input_caps==cfg.EOS_token_id, pad_token_input_caps, input_caps)
        input_cap_masks = input_caps==cfg.PAD_token_id
        
        outputs, _, fake_region = model(samples, input_caps, input_cap_masks)
        
        cap_gen = torch.argmax(outputs[0], dim=-1) # B 
        log_probs = torch.max(F.log_softmax(outputs[0], dim=-1), dim=-1)[0] # compute prediction
        reward = compute_score(cap_gen, caps[:,1:], cfg) # compute reward
        loss = -torch.sum(log_probs, -1) * reward # B 1
        loss1 = loss.mean()
        
        cap_gen = torch.argmax(outputs[1], dim=-1) # B 
        log_probs = torch.max(F.log_softmax(outputs[1], dim=-1), dim=-1)[0] # compute prediction
        reward = compute_score(cap_gen, caps[:,1:], cfg) # compute reward
        loss = -torch.sum(log_probs, -1) * reward # B 1
        loss2 = loss.mean()
        
        cap_gen = torch.argmax(outputs[2], dim=-1) # B 
        log_probs = torch.max(F.log_softmax(outputs[2], dim=-1), dim=-1)[0] # compute prediction
        reward = compute_score(cap_gen, caps[:,1:], cfg) # compute reward
        loss = -torch.sum(log_probs, -1) * reward # B 1
        loss3 = loss.mean()
        
        loss = loss1 + loss2 + loss3
        
        if not math.isfinite(loss):
            print(f'Loss is {loss}, stopping training')
            sys.exit(1)
        
        optimizer.zero_grad()
        loss.backward()
        
        if cfg.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_max_norm)
        optimizer.step()
        loss_logger.update(loss.item(), images.size(0))

        global_step+=1

        #============ tensorboard train log info ============#
        if args.log:
            lossinfo = {
                'Train_Loss': loss.item(),                                                                                                  
                'Train_Loss_avg': loss_logger.avg,                                                                                                  
                    } 
            #============ print the train log info ============# 
            train_dataloader.set_description(
                'lr: {lr:.8f} | loss: {loss:.8f} '.format(
                    loss=loss_logger.avg,
                    lr = current_lr))


def train(args, cfg, train_dataloader, train_sampler, val_dataloader, model, summary_writer, logger, log_dir):
    max_epochs = cfg.epochs
    max_iters = len(train_dataloader[0])
    sum_steps = max_epochs*max_iters

    model, optimizer, start_epoch, scheduler = preset_model(args, cfg, model, logger, sum_steps)
    
    criterion1 = torch.nn.CrossEntropyLoss(ignore_index = cfg.PAD_token_id).cuda(args.gpu)
    criterion2 = torch.nn.MSELoss().cuda(args.gpu)

    global_step = start_epoch * len(train_dataloader[0])
    
    if args.log:
        logger.info(f'global_step: {global_step}...')

    best_val_acc_fixed = 0        
    best_val_acc_adaptive = 0        

    base_lr = cfg.lr
    use_rl = 0
    
    for current_epoch in range(start_epoch, max_epochs):
        loss_logger = AverageMeter()

        if not use_rl:
            current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        else:
            optimizer_rl = torch.optim.AdamW(
                model.parameters(), lr=1e-6, weight_decay=cfg.weight_decay)
            current_lr = optimizer_rl.state_dict()['param_groups'][0]['lr']

        model.train()
        criterion1.train()
        criterion2.train()
        
        if current_epoch < 171:
            if args.log:
                if not use_rl:
                    train_dataloader[0] = tqdm(train_dataloader[0], dynamic_ncols=True)
                    logger.info(f'############# Starting Epoch {current_epoch} | LR: {current_lr} #############')
                else:
                    train_dataloader[0] = tqdm(train_dataloader[0], dynamic_ncols=True)
                    logger.info(f'########### Starting RL Epoch {current_epoch} | LR: {current_lr} ###########')
                    
            train_sampler[0].set_epoch(current_epoch)

            if not use_rl:
                train_ce(args, cfg, model, train_dataloader[0], optimizer, [criterion1, criterion1, criterion1, criterion2], loss_logger, summary_writer, global_step, logger)
            else:
                train_scst(args, cfg, model, train_dataloader[0], optimizer_rl, loss_logger, summary_writer, global_step)
            
        else:
            if args.log:
                train_dataloader[1] = tqdm(train_dataloader[1], dynamic_ncols=True)
                logger.info(f'############# Starting Tune Epoch {current_epoch} | LR: {current_lr} #############')
                
            train_sampler[1].set_epoch(current_epoch)
            train_ce(args, cfg, model, train_dataloader[0], optimizer, [criterion1, criterion1, criterion1, criterion2], loss_logger, summary_writer, global_step, logger)
        
        scheduler.step(current_epoch)
        
        # ----------
        #  Validation
        # ----------
        if ((current_epoch+1) % args.val_epoch == 0 and current_epoch > 60):
            if args.log:
                model_save_path = os.path.join(log_dir, 'snapshots')    
                mkdir(model_save_path) 
                ACC_fixed, ACC_adaptive = evalute_transformer(cfg, val_dataloader, model.module)

                #============ print the val log info ============#
                logger.epochInfo('Validation', current_epoch, 100*ACC_fixed, 100*ACC_adaptive)
                #============ tensorboard val log info ============#
                
                print('Val_AUC_fixed: {}, Val_AUC_adaptive: {}'.format(100 * ACC_fixed, 100 * ACC_adaptive))
                
                if ACC_fixed >= best_val_acc_fixed:
                    best_val_acc_fixed = ACC_fixed
                    torch.save({
                        'best_state_dict_fixed': model.module.state_dict(),
                        }, os.path.join(model_save_path, "best_model_fixed.pt"))

                if ACC_adaptive >= best_val_acc_adaptive:
                    best_val_acc_adaptive = ACC_adaptive
                    torch.save({
                        'best_state_dict_adaptive': model.module.state_dict(),
                        }, os.path.join(model_save_path, "best_model_adaptive.pt"))


def main_worker(gpu, args, cfg):
    if gpu is not None:
        args.gpu = gpu

    init_dist(args)
    
    log_dir = os.path.join(args.results_dir, cfg.backbone, args.dataset_name, args.log_name)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'log.txt')
    logger = setlogger(log_file)
    
    if args.log:
        summary_writer = None
    else:
        summary_writer = None
    
    if args.log:
        logger.info('******************************')
        logger.info(args)
        logger.info('******************************')
        logger.info(cfg.__dict__)
        logger.info('******************************')

    # model
    model = MMNet.build_model(cfg)

    # TODO: check its performance
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()

    batch_size = cfg.batch_size
    
    # facial attributes
    idx_bangs = [68,69,70,71,72,73,75,76,79,80]
    idx_eyeglasses = [0, 16] + list(range(17,27)) + list(range(36, 48)) 
    idx_smiling = list(range(48, 68))
    idx_bread = list(range(5, 11))
    idx_young = list(range(0, 81))
    
    # facial components
    nose = list(range(27, 36))
    eye = list(range(36, 41)) + list(range(42, 48))
    eyebrow = list(range(17, 22)) + list(range(22, 27))
    lip = list(range(48, 68))
    hair = list(range(0,81))

    train_dataset = SeqDeepFakeDataset(
        cfg=cfg,
        mode="train",
        data_root=args.data_dir,
        dataset_name=args.dataset_name,
        # bbox=[idx_bangs, idx_eyeglasses, idx_bread, idx_smiling, idx_young] # for attributes
        bbox = [nose, eye, eyebrow, lip, hair] # for components
    )
    
    tune_dataset = SeqDeepFakeDataset(
        cfg=cfg,
        mode="tune",
        data_root=args.data_dir,
        dataset_name=args.dataset_name
        # bbox=[[[27,35]], [[36,41], [42,47]], [[17,21], [22,26]], [[48,67]], [[0,80]]]
    )
    
    val_dataset = SeqDeepFakeDataset(
        cfg=cfg,
        mode="test",
        data_root=args.data_dir,
        dataset_name=args.dataset_name
    )
    
    if args.log:
        print('train:',len(train_dataset))
        
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=args.world_size, rank=args.rank)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None), num_workers=12, sampler=train_sampler)
    
    tune_sampler = torch.utils.data.distributed.DistributedSampler(
        tune_dataset, num_replicas=args.world_size, rank=args.rank)
    tune_dataloader = torch.utils.data.DataLoader(
        tune_dataset, batch_size=batch_size, shuffle=(tune_sampler is None), num_workers=12, sampler=tune_sampler)
        
    if args.log:
        print('val:',len(val_dataset))
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=12)
    train(args, cfg, [train_dataloader, tune_dataloader], [train_sampler, tune_sampler], val_dataloader, model, summary_writer, logger, log_dir)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--cfg', type=str, default=None, help='path of config json file')
    arg('--results_dir', type=str, default='results')
    arg('--data_dir', type=str, default=None)
    arg('--dataset_name', type=str, default=None)
    arg('--resume', type=str, default=None)
    arg('--log_name', '-l', type=str)

    arg('--model_save_epoch', type=int, default=None)
    arg('--val_epoch', type=int, default=1)
    arg('--manual_seed', type=int, default=777)

    arg('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    arg('--world_size', default=1, type=int,
                        help='world size for distributed training')
    arg('--dist-url', default='tcp://127.0.0.1:23459', type=str,
                        help='url used to set up distributed training')
    arg('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    arg('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none',
                        help='job launcher')

    args = parser.parse_args()
    set_random_seed(args.manual_seed, True)
    cfg = Config(args.cfg)

    if args.launcher == 'none':
        args.launcher = 'pytorch'
        main_worker(0, args, cfg)
    else:
        ngpus_per_node = torch.cuda.device_count()
        args.ngpus_per_node = ngpus_per_node
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(args, cfg))
