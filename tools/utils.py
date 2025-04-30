import torch
from typing import Optional, List
from torch import Tensor
import torch.distributed as dist
import numpy as np
import torch.nn.functional as F
from PIL import Image

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.cuda(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.cuda(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self): # 告诉外界类中属性信息
        return str(self.tensors)


def nested_tensor_from_tensor_list(imgsize, tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = [3, imgsize, imgsize]
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)

        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img) #除图以外其余为0
            m[: img.shape[1], :img.shape[2]] = False #图的区域被置为False

    else:
        raise ValueError('not supported')

    return NestedTensor(tensor, mask)

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def Bbox_generation_naive(landmark_path, bbox_idx, img_size=255):
    '''
    bbox_idx: idx_min : idx_max
    '''
    landmarks = np.load(landmark_path)
    bboxs = []
    
    for i in bbox_idx:
        keys = landmarks[i[0]:i[1]]
        key_max = np.max(keys, axis=0)
        key_min = np.min(keys, axis=0)
        bboxs.append([np.maximum(key_min[0]-5, 0), np.maximum(key_min[1]-5, 0), 
                      np.minimum(key_max[0]+5, img_size), np.minimum(key_max[1]+5, img_size)])
    
    bboxs = np.array(bboxs)
    bboxs = torch.Tensor(bboxs)
    
    return bboxs

def Bbox_generation(landmark_path, bbox_idx, img_size=255):
    '''
    bbox_idx: idx_min : idx_max
    '''
    landmarks = np.load(landmark_path)
    bboxs = []

    for i in bbox_idx:
        keys = landmarks[i]
        key_max = np.max(keys, axis=0)
        key_min = np.min(keys, axis=0)
        bboxs.append([np.maximum(key_min[0], 0), np.maximum(key_min[1], 0), 
                        np.minimum(key_max[0], img_size), np.minimum(key_max[1], img_size)])

    bboxs = np.array(bboxs)
    bboxs = torch.Tensor(bboxs).reshape(5, -1)
    
    return bboxs

def Mask_generation(image, bbox, caption, reduction):
    H, W = image.size(2), image.size(3)
    mask1 = torch.zeros((H, W))
    mask2 = torch.zeros((H, W))
    idx = np.where(caption==6)[0][0]
    cap = caption[1:idx] - 1
    
    for i in cap:
        if i != 4:
            mask1[int(bbox[i][1]): int(bbox[i][3]), int(bbox[i][0]): int(bbox[i][2])] = 1
        else:
            mask2[int(bbox[i][1]): int(bbox[i][3]), int(bbox[i][0]): int(bbox[i][2])] = 0.5
            
    mask = mask1 + mask2
    mask = mask.to(torch.float32).view(1, 1, mask.size(0), mask.size(1))
    mask[mask > 1] = 1.0
    # if 4 in cap:
    #     c1, c2 = mask1.view(1,256,256)[:,:,:,None].expand(-1,-1,-1,3), mask2.view(1,256,256)[:,:,:,None].expand(-1,-1,-1,3)
    #     c1, c2 = c1.cpu().squeeze(0).numpy(), c2.cpu().squeeze(0).numpy()
    #     image_pil=Image.fromarray(np.uint8(image[0].permute(1,2,0) * 255))
    #     image_pil.save("./img.jpg")
    #     image_pil=Image.fromarray(np.uint8(c1*255))
    #     image_pil.save("./c1.jpg")
    #     image_pil=Image.fromarray(np.uint8(c2*255))
    #     image_pil.save("./c2.jpg")
    
    # mask1[mask2 == scale] = scale_s # 
    # mask2[mask2 != scale] = scale_l # bug mask2在不是背景的时候会全变0.1
    
    
    # print(mask1)
    # print(mask1.shape)
    # if 1 in cap and 2 in cap:
    #     c3, c4 = mask1.view(1,16,16)[:,:,:,None].expand(-1,-1,-1,3), mask2.view(1,16,16)[:,:,:,None].expand(-1,-1,-1,3)
    #     c3, c4 = c3.cpu().squeeze(0).numpy(), c4.cpu().squeeze(0).numpy()
    #     image_pil=Image.fromarray(np.uint8(image[0].permute(1,2,0) * 255))
    #     image_pil.save("./img.jpg")
    #     image_pil=Image.fromarray(np.uint8(c3*255))
    #     image_pil.save("./c3.jpg")
    #     image_pil=Image.fromarray(np.uint8(c4*255))
    #     image_pil.save("./c4.jpg")
    
    mask = F.interpolate(mask, (H//reduction, W//reduction), mode='nearest')
    mask = mask[0]
    
    return mask

def Mask_generation_hinge(image, bbox, caption, reduction):
    H, W = image.size(2), image.size(3)
    mask1 = torch.zeros((H, W)).to(torch.bool)
    mask2 = torch.zeros((H, W)).to(torch.bool)
    idx = np.where(caption==6)[0][0]
    cap = caption[1:idx] - 1
    
    for i in cap:
        if i != 4:
            mask1[int(bbox[i][1]):int(bbox[i][3]), int(bbox[i][0]):int(bbox[i][2])] = 1

        else:
            mask2[int(bbox[i][1]):int(bbox[i][3]), int(bbox[i][0]):int(bbox[i][2])] = 1
            mask2 = ~mask2 # 大区域

    mask1 = mask1.to(torch.float32).view(1, 1, mask1.size(0), mask1.size(1))
    mask2 = mask2.to(torch.float32).view(1, 1, mask2.size(0), mask2.size(1))

    mask2[mask2 == 1] = -1
    mask2[mask2 == 0] = 1

    # mask1[mask2 == scale] = scale_s # 
    # mask2[mask2 != scale] = scale_l # bug mask2在不是背景的时候会全变0.1
    
    mask1 = F.interpolate(mask1, (H//reduction, W//reduction), mode='nearest')
    mask2 = F.interpolate(mask2, (H//reduction, W//reduction), mode='nearest')
    mask1 = mask1.flatten(0,2)
    mask2 = mask2.flatten(0,2)
    
    return [mask1, mask2]

# def Mask_generation2(image, bbox, caption, reduction):
#     H, W = image.size(2), image.size(3)
#     mask1 = torch.zeros((H, W)).to(torch.bool)
#     mask2 = torch.zeros((H, W)).to(torch.bool)
#     idx = np.where(caption==6)[0][0]
#     cap = caption[1:idx] - 1
#     scale = 0.75
#     scale_s = 0.0
#     scale_l = 0.0

#     for i in cap:
#         if i != 4:
#             mask1[int(bbox[i][1]):int(bbox[i][3]), int(bbox[i][0]):int(bbox[i][2])] = 1
#             scale_l = 0.1
#         else:
#             mask2[int(bbox[i][1]):int(bbox[i][3]), int(bbox[i][0]):int(bbox[i][2])] = 1
#             mask2 = ~mask2 # 大区域
#             scale_s = 0.1

#     mask1 = mask1.to(torch.float32).view(1, 1, mask1.size(0), mask1.size(1))
#     mask2 = mask2.to(torch.float32).view(1, 1, mask2.size(0), mask2.size(1))
#     mask2 = mask2 * scale
#     mask1[mask2 == scale] = scale_s # mask2 
#     # mask2[:, :, int(bbox[-1][1]) : int(bbox[-1][3]), int(bbox[-1][0]) : int(bbox[-1][2])] = scale_l # mask2[mask2 != scale] = scale_l
#     mask2[mask1 == 1] = scale_l
     
#     mask1 = F.interpolate(mask1, (H//reduction, W//reduction), mode='nearest')
#     mask2 = F.interpolate(mask2, (H//reduction, W//reduction), mode='nearest')
    
#     mask1 = mask1.flatten(0,2)
#     mask2 = mask2.flatten(0,2)
    
#     return [mask1, mask2]
