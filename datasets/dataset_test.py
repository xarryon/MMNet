import torch
from torch.utils.data import Dataset
import numpy as np
import os
import os.path
import random
from PIL import Image
import pandas as pd
from torchvision import transforms
import torchvision.transforms.functional as tf
from tools.utils import nested_tensor_from_tensor_list, Bbox_generation, Mask_generation
from io import BytesIO

def randomJPEGcompression(image):
    qf = random.randrange(60, 100)
    outputIoStream = BytesIO()
    image.save(outputIoStream, "JPEG", quality = qf, optimice = True)
    outputIoStream.seek(0)
    return Image.open(outputIoStream)

def read_data(file):
    info = pd.read_csv(file)
    img_list = info['file_path'].tolist()
    label_list = info['label'].tolist()
    return img_list, label_list

def make_dataset(csv_file, root=None):
    dataset = []

    imgs, labels = read_data(csv_file)

    for i in range(len(imgs)):
        if root:
            imgs[i] = os.path.join(root, imgs[i]) # from relative path to absolute path
        dataset.append((imgs[i], labels[i]))

    return dataset


def create_train_transforms(image_size):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.Lambda(randomJPEGcompression),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def create_tune_transforms(image_size):
    return transforms.Compose([
        transforms.Resize(image_size//4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    

def create_val_transforms(image_size):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


class SeqDeepFakeDataset(Dataset):

    def __init__(self,
                 cfg=None,
                 data_root=None,
                 mode="train",
                 dataset_name=None,
                 bbox=None
                 ):
        super().__init__()
        self.mode = mode
        self.cfg = cfg
        if self.mode == "train":
            self.transforms = create_train_transforms(cfg.imgsize)
            self.transforms_gt = create_tune_transforms(cfg.imgsize)
        elif self.mode == 'tune':
            self.transforms = create_tune_transforms(cfg.imgsize)
            self.transforms_gt = create_tune_transforms(cfg.imgsize)
        elif self.mode in ["val", "test"]:
            self.transforms = create_val_transforms(cfg.imgsize)
            self.transforms_gt = create_val_transforms(cfg.imgsize)
        else:
            raise ValueError(f"WRONG INPUT MODE: {self.mode}!")

        if self.mode == 'tune':
            mode = 'train'
            
        self.data = make_dataset(os.path.join(data_root, f"{dataset_name}/annotations/{mode}.csv"), root=data_root)
        self.bbox = bbox
        
        self.SOS_token_id = cfg.SOS_token_id
        self.EOS_token_id = cfg.EOS_token_id
        self.PAD_token_id = cfg.PAD_token_id

    def __getitem__(self, index: int):
        img_path, label = self.data[index]
        
        if self.mode in ["train", "tune"]:
            label_list_split = label.split(',')

            label_list = [int(label_list_split[0][1]), int(label_list_split[1][1]), int(label_list_split[2][1]), int(label_list_split[3][1]), int(label_list_split[4][1])]
            caption = np.array(label_list)
            
            if np.count_nonzero(caption) == 0:
                caption_mask = (1-np.array([1, 1, 0, 0, 0, 0, 0])).astype(bool)
                caption = np.insert(caption, [0, 0], [self.SOS_token_id, self.EOS_token_id])
                caption[np.where(caption_mask==True)] = self.PAD_token_id

            elif np.count_nonzero(caption) == len(caption):
                caption_mask = (1-np.array([1, 1, 1, 1, 1, 1, 1])).astype(bool)
                caption = np.insert(caption, [0, 5], [self.SOS_token_id, self.EOS_token_id])
            else:
                first_zero_idx = np.where(caption==0)[0][0]
                caption = np.insert(caption, [0, first_zero_idx], [self.SOS_token_id, self.EOS_token_id])
                EOS_idx = np.where(caption==self.EOS_token_id)[0][0]
                caption_mask = (1-np.pad(np.ones(EOS_idx+1), (0, len(caption)-(EOS_idx+1)))).astype(bool)
                caption[np.where(caption_mask==True)] = self.PAD_token_id
            
            caption_reverse = np.ones_like(caption) * 7
            caption_reverse[0] = 0
            cnt = 1
            for i in range(len(caption)):
                if caption[len(caption)-i-1] != self.PAD_token_id and caption[len(caption)-i-1] != self.SOS_token_id and caption[len(caption)-i-1] != self.EOS_token_id :
                    caption_reverse[cnt] = caption[len(caption)-1-i]
                    cnt += 1
            if cnt != 6:
                caption_reverse[cnt] = self.EOS_token_id
            else:
                caption_reverse[-1] = self.EOS_token_id
                
            image = Image.open(img_path).convert('RGB')

            if self.transforms:
                image = self.transforms(image)

            image = nested_tensor_from_tensor_list(self.cfg.imgsize, image.unsqueeze(0))
            
            length = torch.sum(~torch.from_numpy(caption_mask[1:])) - 1

            seqlabel = int(''.join(map(str, label_list)))
            
            flip_label = random.random() > 0.5
            
            if self.bbox is not None:
                self.bbox = np.array(self.bbox)
                assert self.bbox.shape[0] == 5, 'region numbers can not compatiable'
                
                landmark_path = img_path
                for old, new in ([['images', 'landmarks'], ['.jpg', '.npy']]):
                    landmark_path =  landmark_path.replace(old, new)
                landmark_path =  landmark_path.replace('.png', '.npy')
                
                bbox = Bbox_generation(landmark_path, self.bbox)
                gt_region = Mask_generation(image.tensors, bbox, caption, reduction=4)
                
                if flip_label:
                    gt_region = tf.hflip(gt_region)
                    image.tensors = tf.hflip(image.tensors)
                
                return image.tensors.squeeze(0), image.mask.squeeze(0), caption, caption_mask, length, seqlabel, gt_region
            
            else:
                gt_idx = img_path[::-1].index('/')
                gt_path = os.path.join('./real_imgs/celeba_hq/trainval', img_path[-gt_idx:])
                gt_img = Image.open(gt_path).convert('RGB')
                gt_img = self.transforms_gt(gt_img)
                
                if flip_label:
                    gt_img = tf.hflip(gt_img)
                    image.tensors = tf.hflip(image.tensors)

                return image.tensors.squeeze(0), image.mask.squeeze(0), caption, caption_mask, length, seqlabel, gt_img
        
        elif self.mode in ["val", "test"]:
            label_list_split = label.split(',')
            label_list = [int(label_list_split[0][1]), int(label_list_split[1][1]), int(label_list_split[2][1]), int(label_list_split[3][1]), int(label_list_split[4][1])]
            image = Image.open(img_path).convert('RGB')
            
            if self.transforms:
                image = self.transforms(image)
            
            return image, torch.FloatTensor(label_list)
        else:
            raise ValueError(f"WRONG INPUT MODE: {self.mode}!")

    def __len__(self):
        return len(self.data)
