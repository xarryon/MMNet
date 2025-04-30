import torch
import torch.nn as nn
import torch.nn.functional as F

def con_loss_extra(features, labels, gpu=True):
    B, _ = features.shape
    features = F.normalize(features)
    cos_matrix = features.mm(features.t())
    pos_label_matrix = torch.stack([labels == labels[i] for i in range(B)]).float()
    neg_label_matrix = 1 - pos_label_matrix
    pos_cos_matrix = 1 - cos_matrix
    neg_cos_matrix = cos_matrix - 0.7
    neg_cos_matrix[neg_cos_matrix < 0] = 0
    
    if gpu == True:
        pos_label_matrix = pos_label_matrix.cuda()
        neg_label_matrix = neg_label_matrix.cuda()
    
    loss = (pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum()
    loss /= (B * B)

    return loss

def con_loss_inner(features, labels, inner_label, gpu=True):
    loss = 0

    for i in range(1, 6):
        sub_features = features[labels == i]
        sub_inner_label = inner_label[labels == i]

        B, _ = sub_features.shape
        if B == 0:
            continue
        sub_features = F.normalize(sub_features)
        cos_matrix = sub_features.mm(sub_features.t())
        pos_label_matrix = torch.stack([sub_inner_label == sub_inner_label[i] for i in range(B)]).float()
        neg_label_matrix = 1 - pos_label_matrix
        pos_cos_matrix = 1 - cos_matrix
        neg_cos_matrix = cos_matrix - 0.8
        neg_cos_matrix[neg_cos_matrix < 0] = 0
        
        if gpu == True:
            pos_label_matrix = pos_label_matrix.cuda()
            neg_label_matrix = neg_label_matrix.cuda()
        
        loss += ((pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum()) / (B * B)

    return loss


def con_loss_extra_m(features, labels):
    interval_list = torch.Tensor([[1, 0.7, 0.6, 0.5, 0.4, 0.3],
                                  [0.7, 1, 0.7, 0.6, 0.5, 0.4],
                                  [0.6, 0.7, 1, 0.6, 0.5, 0.4],
                                  [0.5, 0.6, 0.7, 1, 0.7, 0.6],
                                  [0.4, 0.5, 0.6, 0.7, 1, 0.7],
                                  [0.3, 0.4, 0.5, 0.6, 0.7, 1]]).cuda()
    
    interval_list_p = torch.Tensor([[1, 0, 0, 0, 0, 0],
                                    [0, 0.9, 0, 0, 0, 0],
                                    [0, 0, 0.9, 0, 0, 0],
                                    [0, 0, 0, 0.9, 0, 0],
                                    [0, 0, 0, 0, 0.9, 0],
                                    [0, 0, 0, 0, 0, 0.9]]).cuda()
    
    # interval_list = torch.Tensor([[1, 0.7, 0.6, 0.5, 0.4, 0.3],
    #                               [0.7, 1, 0.7, 0.6, 0.5, 0.4],
    #                               [0.6, 0.7, 1, 0.7, 0.6, 0.5],
    #                               [0.5, 0.6, 0.7, 1, 0.7, 0.6],
    #                               [0.4, 0.5, 0.6, 0.7, 1, 0.7],
    #                               [0.3, 0.4, 0.5, 0.6, 0.7, 1]]).cuda()
    
    # interval_list_p = torch.Tensor([[0.7, 0, 0, 0, 0, 0],
    #                                 [0, 0.7, 0, 0, 0, 0],
    #                                 [0, 0, 0.7, 0, 0, 0],
    #                                 [0, 0, 0, 0.7, 0, 0],
    #                                 [0, 0, 0, 0, 0.7, 0],
    #                                 [0, 0, 0, 0, 0, 0.7]]).cuda()
    
    B, _ = features.shape
    features = F.normalize(features)
    cos_matrix = features.mm(features.t())
    x, y = torch.meshgrid(labels, labels)
    interval_matrix = interval_list[x, y]
    interval_matrix_p = interval_list_p[x, y]
    
    neg_cos_matrix = cos_matrix - interval_matrix
    neg_cos_matrix[neg_cos_matrix < 0] = 0
    
    pos_cos_matrix = interval_matrix_p - cos_matrix
    pos_cos_matrix[pos_cos_matrix < 0] = 0
    
    loss = neg_cos_matrix.sum() + pos_cos_matrix.sum()
    
    # loss = neg_cos_matrix.sum()
    # loss = pos_cos_matrix.sum()
    
    loss /= (B * B)

    return loss


def con_loss_m2(features, labels):
    interval_list_l = torch.Tensor([[0.9, 0.7, 0.5, 0.3, 0.1, 0.0],
                                 [0.7, 0.9, 0.7, 0.5, 0.3, 0.1],
                                 [0.5, 0.7, 0.9, 0.7, 0.5, 0.3],
                                 [0.3, 0.5, 0.7, 0.9, 0.7, 0.5],
                                 [0.1, 0.3, 0.5, 0.7, 0.9, 0.7],
                                 [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]]).cuda()
    
    interval_list_h = torch.Tensor([[1, 0.9, 0.7, 0.5, 0.3, 0.1],
                                 [0.9, 1, 0.9, 0.7, 0.5, 0.3],
                                 [0.7, 0.9, 1, 0.9, 0.7, 0.5],
                                 [0.5, 0.7, 0.9, 1, 0.9, 0.7],
                                 [0.3, 0.5, 0.7, 0.9, 1, 0.9],
                                 [0.1, 0.3, 0.5, 0.7, 0.9, 1]]).cuda()
    
    
    B, _ = features.shape
    features = F.normalize(features)
    cos_matrix = features.mm(features.t())
    x, y = torch.meshgrid(labels, labels)
    interval_matrix_l = interval_list_l[x, y]
    interval_matrix_h = interval_list_h[x, y]
    
    low_cos_matrix = cos_matrix - interval_matrix_l
    high_cos_matrix = interval_matrix_h - cos_matrix
    low_cos_matrix[low_cos_matrix > 0] = 0
    high_cos_matrix[high_cos_matrix > 0] = 0
    low_cos_matrix = -low_cos_matrix
    high_cos_matrix = -high_cos_matrix
    
    loss = low_cos_matrix.sum() + high_cos_matrix.sum()
    loss /= (B * B)
    # loss /= B

    return loss


class focal_loss(nn.Module):
    def __init__(self, region=None, alpha=1, gamma=0, ignore_idx=7, size_average=True):
        super(focal_loss, self).__init__()
        self.size_average = size_average
        self.ignore = ignore_idx
        self.alpha = torch.ones(8)
        self.alpha[ignore_idx] = 0
        
        if region is not None:
            if region == 's':
                self.alpha[[1, 4, 5]] = 0.9
                self.alpha[[2, 3]] = 1.15
            elif region == 'm':
                self.alpha[[2, 3, 5]] = 0.9
                self.alpha[[1, 4]] = 1.15
            elif region == 'l':
                self.alpha[[1, 2, 3, 4]] = 0.9
                self.alpha[5] = 1.3
                
        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B, C, D]  分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B, D]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        B, C, D = preds.shape
        N = B * D - len(torch.where(labels.contiguous().view(-1)==self.ignore)[0])
        self.alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1) # log_softmax
        preds_softmax = torch.exp(preds_logsoft)    # softmax
        
        preds_softmax = preds_softmax.gather(1, labels[:, None, :]).view(-1, 1)   # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1, labels[:, None, :]).view(-1, 1)
        alphas = self.alpha.gather(0, labels.contiguous().view(-1)).view(B, D).view(-1, 1)

        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = torch.mul(alphas, loss).view(B, D)
        
        if self.size_average:
            loss = loss.sum()/N
        else:
            loss = loss.sum()
            
        return loss

def con_loss_each(features, gpu=True):
    f1, f2, f3 = features[0].view(-1,256), features[1].view(-1,256), features[2].view(-1,256)
    f1 = F.normalize(f1)
    f2 = F.normalize(f2)
    f3 = F.normalize(f3)
    eye = torch.eye(f1.shape[0]).cuda()
    
    cos_matrix1 = f1.mm(f2.t())
    cos_matrix2 = f2.mm(f3.t())
    cos_matrix3 = f1.mm(f3.t())
    
    cos_matrix1 = torch.abs(cos_matrix1*eye)
    cos_matrix2 = torch.abs(cos_matrix2*eye)
    cos_matrix3 = torch.abs(cos_matrix3*eye)
 
    loss = torch.sum(cos_matrix1) + torch.sum(cos_matrix2) + torch.sum(cos_matrix3)

    return loss


if __name__ == "__main__":
    a = torch.randn(1, 8, 3)
    l = torch.Tensor([[1, 7, 2]])
    print(l.shape)
    l = l.long()
    ce = torch.nn.CrossEntropyLoss(ignore_index=7)
    loss = ce(a,l)
    f = focal_loss(region='s')
    loss2 = f(a,l)
    print(loss)
    print(loss2)
    # ms = torch.meshgrid(l, l)
    # interval_dis = torch.Tensor([[0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
    #                              [0.8, 0.9, 0.8, 0.7, 0.6, 0.5],
    #                              [0.7, 0.8, 0.9, 0.8, 0.7, 0.6],
    #                              [0.6, 0.7, 0.8, 0.9, 0.8, 0.7],
    #                              [0.5, 0.6, 0.7, 0.8, 0.9, 0.8],
    #                              [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]])
    
    # interval_list_p = torch.Tensor([[0.8, 0, 0, 0, 0, 0],
    #                                 [0, 0.8, 0, 0, 0, 0],
    #                                 [0, 0, 0.8, 0, 0, 0],
    #                                 [0, 0, 0, 0.8, 0, 0],
    #                                 [0, 0, 0, 0, 0.8, 0],
    #                                 [0, 0, 0, 0, 0, 0.8]])
    # x = ms[0].to(torch.long)
    # y = ms[1].to(torch.long)
    # print(interval_dis[x, y])
    # print(interval_list_p[x, y])