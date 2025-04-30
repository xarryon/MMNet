from builtins import len, print
import torch

time_list = {1:torch.LongTensor([[i] for i in range(5)]),
             2:torch.LongTensor([[i, i+1] for i in range(4)]),
             3:torch.LongTensor([[i, i+1, i+2] for i in range(3)]),
             4:torch.LongTensor([[i, i+1, i+2, i+3] for i in range(2)]),
             5:torch.LongTensor([[0, 1, 2, 3, 4]])}


def multiTimeAdaPrecision_cpu(time, preds, gts):
    cnt = 0
    correct = 0
    preds = preds[preds>0]
    gts = gts[gts>0]
    delta = len(preds) - len(gts)
    preds = preds
    gts = gts
    
    if delta < 0:
        preds = torch.cat((preds, torch.zeros(-delta)), dim=0)
    elif delta > 0:
        gts = torch.cat((gts, torch.zeros(delta)), dim=0)
    
    if (len(preds) == 0 and len(gts) == 0) or len(preds) < time:
        return -1
 
    for t in time_list[time]:
        if t[-1] + 1 > len(gts):
            break
        pred = torch.index_select(preds, 0, t)
        gt = torch.index_select(gts, 0, t)
        if torch.sum(pred == gt) == time:
            correct += 1
        cnt += 1

    mtada = correct / cnt
    
    return mtada

def multiTimeAdaRecall_cpu(time, preds, gts):
    cnt = 0
    correct = 0
    preds = preds[preds>0]
    gts = gts[gts>0]
    delta = len(preds) - len(gts)
    preds = preds
    gts = gts
    
    if delta < 0:
        preds = torch.cat((preds, torch.zeros(-delta)), dim=0)

    if (len(preds) == 0 and len(gts) == 0) or len(gts) < time:
        return -1
        
    for t in time_list[time]:
        if t[-1] + 1 > len(gts):
            break
        pred = torch.index_select(preds, 0, t)
        gt = torch.index_select(gts, 0, t)
        
        if torch.sum(pred == gt) == time:
            correct += 1
        cnt += 1

    mtada = correct / cnt
    
    return mtada


def Recover_evaluation(preds, gts):
    preds = preds[preds>0]
    gts = gts[gts>0]
    delta = len(preds) - len(gts)
    preds = preds.to(torch.float)
    gts = gts.to(torch.float)
    
    if (len(preds) == 0 and len(gts) == 0):
        return -1
 
    if delta != 0:
        return 0
    
    if torch.sum(preds == gts) == len(gts):
        return 1
    else:
        return 0


def multiTimeAdaPrecision(time, preds, gts):
    cnt = 0
    correct = 0
    preds = preds[preds>0]
    gts = gts[gts>0]
    delta = len(preds) - len(gts)
    preds = preds.to(torch.float)
    gts = gts.to(torch.float)
    
    if delta < 0:
        # preds = torch.cat((preds, torch.zeros(-delta).cuda()), dim=0)
        preds = preds[:len(gts)]
    elif delta > 0:
        # gts = torch.cat((gts, torch.zeros(delta).cuda()), dim=0)
        gts = gts[:len(preds)]
        
    if (len(preds) == 0 and len(gts) == 0) or len(preds) < time:
        return -1
        
    for t in time_list[time].cuda():
        if t[-1] + 1 > len(gts):
            break
        pred = torch.index_select(preds, 0, t)
        gt = torch.index_select(gts, 0, t)
        
        if torch.sum(pred == gt) == time:
            correct += 1
        cnt += 1

    mtada = correct / cnt
    
    return mtada


def multiTimeAdaRecall(time, preds, gts):
    cnt = 0
    correct = 0
    preds = preds[preds>0]
    gts = gts[gts>0]
    delta = len(preds) - len(gts)
    preds = preds.to(torch.float)
    gts = gts.to(torch.float)
    
    if delta < 0:
        preds = torch.cat((preds, torch.zeros(-delta).cuda()), dim=0)

    if (len(preds) == 0 and len(gts) == 0) or len(gts) < time:
        return -1
        
    for t in time_list[time].cuda():
        if t[-1] + 1 > len(gts):
            break
        pred = torch.index_select(preds, 0, t)
        gt = torch.index_select(gts, 0, t)
        
        if torch.sum(pred == gt) == time:
            correct += 1
        cnt += 1

    mtada = correct / cnt
    
    return mtada