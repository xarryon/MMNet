import os
from pathlib import Path
import sys
import argparse
import cv2
import torch
import numpy as np
import random
import logging
from types import MethodType
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from tools.env import init_dist
import torch.multiprocessing as mp
from tqdm import tqdm
sys.path.append(str(Path(__file__).resolve().parents[1]))
from models import mmnet as MMNet
import torch.nn as nn
from sklearn.metrics import roc_auc_score

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

class AccMeter(object):
    def __init__(self):
        self.nums = 0
        self.acc = 0

    def reset(self):
        self.nums = 0
        self.acc = 0

    def update(self, pred, target, use_bce=False):
        if use_bce:
            pred = (pred >= 0.5).int()
        else:
            pred = pred.argmax(1)
        self.nums += target.shape[0]
        self.acc += torch.sum(pred == target)

    def mean_acc(self):
        return self.acc / self.nums

class AUCMeter(object):
    def __init__(self):
        self.score = None
        self.true = None

    def reset(self):
        self.score = None
        self.true = None

    def update(self, score, true, use_bce=False):
        if use_bce:
            score = score.detach().cpu().numpy()
        else:
            score = torch.softmax(score.detach(), dim=-1)
            score = torch.select(score, 1, 1).cpu().numpy()
        true = true.flatten().cpu().numpy()
        self.score = score if self.score is None else np.concatenate([self.score, score])
        self.true = true if self.true is None else np.concatenate([self.true, true])
        
    def mean_auc(self):
        return roc_auc_score(self.true, self.score)
    
def setlogger(log_file):
    filehandler = logging.FileHandler(log_file)
    streamhandler = logging.StreamHandler()

    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)

    def test_acc(self, acc):
        self.info('acc:{acc:.4f}%'.format(
            acc=acc
        ))

    logger.test_acc = MethodType(test_acc, logger)

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


def preset_model(args, cfg, model, logger, test_type):
    if args.ckpt is not None:
        checkpoint_dir = os.path.join(args.results_dir, cfg.backbone, args.dataset_name, args.log_name, 'snapshots', args.ckpt)
    elif test_type == 'fixed':
        checkpoint_dir = os.path.join(args.results_dir, cfg.backbone, args.dataset_name, args.log_name, 'snapshots', 'best_model_fixed.pt')
    elif test_type == 'adaptive':
        checkpoint_dir = os.path.join(args.results_dir, cfg.backbone, args.dataset_name, args.log_name, 'snapshots', 'best_model_adaptive.pt')
        
    checkpoint = torch.load(checkpoint_dir, map_location='cpu')
    
    if args.ckpt is not None:
        model.load_state_dict(checkpoint['state_dict'])
    elif test_type == 'fixed':
        model.load_state_dict(checkpoint['best_state_dict_fixed'])
    elif test_type == 'adaptive':
        model.load_state_dict(checkpoint['best_state_dict_adaptive'])
        
    model.cuda(args.gpu)
    
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)


    return model


def torch2numpy(image):
    image = 0.5 + 0.5 * image
    image = image[0].detach().permute(1,2,0).cpu().numpy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def read_csv(field, file):
    info = pd.read_csv(file)
    image_list = info[field[0]].tolist()
    score_list = info[field[1]].tolist()

    return image_list, score_list


def create_caption_and_mask(cfg):
    caption_template = cfg.PAD_token_id*torch.ones((1, cfg.max_position_embeddings), dtype=torch.long).cuda()
    mask_template = torch.ones((1, cfg.max_position_embeddings), dtype=torch.bool).cuda()

    caption_template[:, 0] = cfg.SOS_token_id
    mask_template[:, 0] = False

    return caption_template, mask_template


def evalute_transformer(cfg, val_dataloader, model, test_type):
    model.eval()
    
    with torch.no_grad():
        running_corrects = 0.0
        epoch_size = 0.0 
        
        for steps, (image, labels) in enumerate(tqdm(val_dataloader)):
            caption, cap_mask = create_caption_and_mask(cfg)
            image, labels = image.cuda(), labels.long().cuda()
            
            prediction_t0 = 0            
            
            for i in range(cfg.max_position_embeddings - 1):
                predictions, heatmaps, rec_ins = model(image, caption, cap_mask)
                
                predictions1 = nn.Softmax(dim=1)(predictions[0][:, i, :]) 
                predictions2 = nn.Softmax(dim=1)(predictions[1][:, i, :]) 
                predictions3 = nn.Softmax(dim=1)(predictions[2][:, i, :]) 
                
                score1 = torch.max(predictions1, dim=-1)[0]
                score2 = torch.max(predictions2, dim=-1)[0]
                score3 = torch.max(predictions3, dim=-1)[0]
                
                idx = torch.argmax(torch.Tensor([score1, score2, score3]))
                                
                if i == 0: # binary deepfake detection
                    prediction_t0 = predictions[idx][:, i, :]
                    prediction_t0 = torch.softmax(prediction_t0, dim=-1)
                    prediction_t0 = torch.Tensor([[prediction_t0[0, 6], 1 - prediction_t0[0, 6]]]).cuda()
                    Y = 0 if torch.all(labels==0) else 1
                    Y = torch.Tensor([Y]).cuda()
                
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

            if test_type == 'fixed':
                running_corrects += torch.sum(caption.cpu() == labels.data.cpu())
                epoch_size += image.size(0) * labels.shape[1]

            elif test_type == 'adaptive':
                cmp_len = max(len(torch.where(labels[0]>0)[0]), len(torch.where(caption[0]>0)[0]))
                if cmp_len == 0:
                    running_corrects += 1
                    cmp_len = 1
                else:
                    running_corrects += torch.sum(caption[:, :cmp_len].cpu() == labels[:, :cmp_len].data.cpu())
                
                epoch_size += image.size(0) * cmp_len
            
    ACC =  running_corrects.double() / epoch_size

    return ACC


def test(args, cfg, test_dataloader, model, logger):
    test_type = args.test_type
    model  = preset_model(args, cfg, model, logger, test_type)
    
    ACC = evalute_transformer(cfg, test_dataloader, model, test_type)
    
    logger.test_acc(100 * ACC)


def main_worker(gpu, args, cfg):
    if gpu is not None:
        args.gpu = gpu

    init_dist(args)

    eval_log_name = 'evaluation.txt'

    model = MMNet.build_model(cfg)
    log_dir = os.path.join(args.results_dir, cfg.backbone, args.dataset_name, args.log_name, eval_log_name)


    logger = setlogger(log_dir)

    logger = logging.getLogger('')
    
    if args.log:
        logger.info('******************************')
        logger.info(args)
        logger.info('******************************')
        logger.info(cfg.__dict__)
        logger.info('******************************')

    from datasets.dataset_test import SeqDeepFakeDataset
    test_dataset = SeqDeepFakeDataset(
        cfg=cfg,
        mode="test",
        data_root=args.data_dir,
        dataset_name=args.dataset_name,
    )

        
    if args.log:
        print('test dataset size:',len(test_dataset))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)
    
    test(args, cfg, test_dataloader, model, logger)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--cfg', type=str, default=None, help='path of config json file')
    arg('--results_dir', type=str, default='results')
    arg('--dataset_name', type=str, default=None)
    arg('--test_type', type=str, default=None)
    arg('--data_dir', type=str, default=None)
    arg('--log_name', '-l', type=str)
    arg('--ckpt', type=str, default=None)

    arg("--padding-part", default=3, type=int)
    arg('--label-smoothing', type=float, default=0.01)

    arg('--manual_seed', type=int, default=777)
    arg('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    arg('--world_size', default=1, type=int,
                        help='world size for distributed training')
    arg('--dist-url', default='tcp://127.0.0.1:23456', type=str,
                        help='url used to set up distributed training')
    arg('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    arg('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none',
                        help='job launcher')

    args = parser.parse_args()
    set_random_seed(args.manual_seed)

    from models.configuration import Config
    cfg = Config(args.cfg)

    if args.launcher == 'none':
        args.launcher = 'pytorch'
        main_worker(0, args, cfg)
    else:
        ngpus_per_node = torch.cuda.device_count()
        args.ngpus_per_node = ngpus_per_node
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(args, cfg))