import torch
import torch.nn.functional as F
import torch.nn as nn

class BeamSearch(object):
    def __init__(self, model, max_len: int, eos_idx: int, beam_size: int):
        self.model = model
        self.max_len = max_len #20
        self.eos_idx = eos_idx #text_field.vocab.stoi['<eos>'] 
        self.beam_size = beam_size #5
        self.b_s = None
        self.device = None
        self.seq_mask = None
        self.seq_logprob = None
        self.outputs = None
        self.log_probs = None
        self.selected_words = None
        self.all_log_probs = None

    def _expand_state(self, selected_beam, cur_beam_size):
        def fn(s):
            shape = [int(sh) for sh in s.shape]
            beam = selected_beam
            for _ in shape[1:]:
                beam = beam.unsqueeze(-1)
            s = torch.gather(s.view(*([self.b_s, cur_beam_size] + shape[1:])), 1,
                             beam.expand(*([self.b_s, self.beam_size] + shape[1:])))
            s = s.view(*([-1, ] + shape[1:]))
            return s

        return fn

    def _expand_visual(self, visual, cur_beam_size: int, selected_beam: torch.Tensor):
        if isinstance(visual, torch.Tensor):
            visual_shape = visual.shape
            visual_exp_shape = (self.b_s, cur_beam_size) + visual_shape[1:]
            visual_red_shape = (self.b_s * self.beam_size,) + visual_shape[1:]
            selected_beam_red_size = (self.b_s, self.beam_size) + tuple(1 for _ in range(len(visual_exp_shape) - 2))
            selected_beam_exp_size = (self.b_s, self.beam_size) + visual_exp_shape[2:]
            visual_exp = visual.view(visual_exp_shape)
            selected_beam_exp = selected_beam.view(selected_beam_red_size).expand(selected_beam_exp_size)
            visual = torch.gather(visual_exp, 1, selected_beam_exp).view(visual_red_shape)
        else:
            new_visual = []
            for im in visual:
                visual_shape = im.shape
                visual_exp_shape = (self.b_s, cur_beam_size) + visual_shape[1:]
                visual_red_shape = (self.b_s * self.beam_size,) + visual_shape[1:]
                selected_beam_red_size = (self.b_s, self.beam_size) + tuple(1 for _ in range(len(visual_exp_shape) - 2))
                selected_beam_exp_size = (self.b_s, self.beam_size) + visual_exp_shape[2:]
                visual_exp = im.view(visual_exp_shape)
                selected_beam_exp = selected_beam.view(selected_beam_red_size).expand(selected_beam_exp_size)
                new_im = torch.gather(visual_exp, 1, selected_beam_exp).view(visual_red_shape)
                new_visual.append(new_im)
            visual = tuple(new_visual)
        return visual

    def apply(self, visual, cap, cap_mask, out_size=1, return_probs=False, **kwargs):
        # 输出经过从大到小排序后的前out_size个outputs
        # images out_size=1
        self.b_s = visual.size(0)
        self.device = visual.device
        self.seq_mask = torch.ones((self.b_s, self.beam_size, 1), device=self.device) # B bs 1
        self.seq_logprob = torch.zeros((self.b_s, 1, 1), device=self.device) # bs 1 1
        self.log_probs = []
        self.selected_words = None
        
        self.cap = cap[:, None, :].expand(-1, self.beam_size, -1)
        self.cap_mask = cap_mask[:, None, :].expand(-1, self.beam_size, -1)
        
        outputs = []
        
        # with self.model.statefulness(self.b_s):#??
        for t in range(self.max_len):
            visual, outputs = self.iter(t, visual, outputs, return_probs, **kwargs)
            
        # Sort result
        # self.seq_logprob = self.seq_logprob - torch.sum(torch.cat(self.log_probs, -1)[..., 4:], dim = 2)[..., None] ## truncation
        
        seq_logprob, sort_idxs = torch.sort(self.seq_logprob, 1, descending=True) # self.seq_logprob 已经排好序
        outputs = torch.cat(outputs, -1) # N bs max_length
        outputs = torch.gather(outputs, 1, sort_idxs.expand(self.b_s, self.beam_size, self.max_len))
        # 根据sort_idxs对预测的句子进行重新排序
        
        log_probs = torch.cat(self.log_probs, -1)
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(self.b_s, self.beam_size, self.max_len))
        # 根据sort_idxs对预测的分数进行重新排序

        outputs = outputs.contiguous()
        log_probs = log_probs.contiguous()

        outputs = outputs[:, :out_size]
        log_probs = log_probs[:, :out_size]
        if out_size == 1:
            outputs = outputs.squeeze(1)
            log_probs = log_probs.squeeze(1)

        return outputs, log_probs

    def select(self, t, candidate_logprob, **kwargs):
        selected_logprob, selected_idx = torch.sort(candidate_logprob.view(self.b_s, -1), -1, descending=True)
        selected_logprob, selected_idx = selected_logprob[:, :self.beam_size], selected_idx[:, :self.beam_size]
        return selected_idx, selected_logprob

    def iter(self, t: int, visual, outputs, return_probs, **kwargs):
        '''
        1.获取当前时刻下所有被选择的beam的所有单词的预测结果
        2.将所有单词结果与对应的beam进行叠加后排序,并获取前beam个分数对应的词的索引
        3.将当前时刻单词结果append至列表后输出
        '''
        # word_logprob 当前预测的单词结果 
        # self.selected_words (b_s 1) 之前不同beam预测的最后词 *输给模型进行预测*
        # self.seq_logprob (N b_s 1) 之前不同beam预测的结果 *需要对其排序*
        # self.log_probs 显示之前每个时刻选择的词的分数
        # candidate_logprob (N bs v) 当前结果与之前结果之和
        cur_beam_size = 1 if t == 0 else self.beam_size # t=0时只需要从SOS的输出中选择前beam个词

        word_logprob = self.step(t, self.model, visual, self.cap, self.cap_mask, outputs)
        # step得到的是decoder的输出
        # t=0:输入图像至encoder后再将结果给decoder，得到decoder的输出结果
        # t>0:输入之前的选择的words给decoder，得到decoder的输出结果
        word_logprob = word_logprob.view(self.b_s, cur_beam_size, -1) # N bs v
        candidate_logprob = self.seq_logprob + word_logprob # N bs v
        
        # Mask sequence if it reaches EOS
        if t > 0:
            mask = (self.selected_words.view(self.b_s, cur_beam_size) != self.eos_idx).float().unsqueeze(-1)
            self.seq_mask = self.seq_mask * mask # 0代表mask
            word_logprob = word_logprob * self.seq_mask.expand_as(word_logprob) # 将EOS后预测结果全部置零
            
            old_seq_logprob = self.seq_logprob.expand_as(candidate_logprob).contiguous() # word_logprob是当前时刻所有预测词结果 old_seq_logprob(N bs v)是当前及之前时刻的预测结果
            old_seq_logprob[:, :, 1:] = -999 # 保留EOS之前的结果并继续参与比较
            candidate_logprob = self.seq_mask * candidate_logprob + old_seq_logprob * (1 - self.seq_mask) 
            # NoEOS:candidate_logprob EOS: old_seq_logprob
        
        # selected_idx, selected_logprob：bs与v合并后进行排序下的 索引 及 对应的分数
        # selected_beam (N bs), selected_words (N bs): 第几个beam 及 对应该beam下的第几个词
        selected_idx, selected_logprob = self.select(t, candidate_logprob, **kwargs) # 将bs与v进行合并从大到小排序并选择前beam_size个
        selected_beam = selected_idx // candidate_logprob.shape[-1]
        selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1] 
        
        # self.model.apply_to_states(self._expand_state(selected_beam, cur_beam_size)) # ??
        # visual = self._expand_visual(visual, cur_beam_size, selected_beam) # bs HW c
        self.seq_logprob = selected_logprob.unsqueeze(-1) 
        self.seq_mask = torch.gather(self.seq_mask, 1, selected_beam.unsqueeze(-1)) # seq_mask N bs 1 selected_beam N bs
        outputs = list(torch.gather(o, 1, selected_beam.unsqueeze(-1)) for o in outputs) # 之前时刻加入的词将作为下一时刻的beam
        outputs.append(selected_words.unsqueeze(-1)) # 加入当前时刻选择的词

        if return_probs:
            if t == 0:
                self.all_log_probs.append(word_logprob.expand((self.b_s, self.beam_size, -1)).unsqueeze(2))
            else:
                self.all_log_probs.append(word_logprob.unsqueeze(2))

        # 利用 selected_beam 获取被选择的beam对应的word_logprob
        this_word_logprob = torch.gather(word_logprob, 1,
                                         selected_beam.unsqueeze(-1).expand(self.b_s, self.beam_size,
                                                                            word_logprob.shape[-1])) # N bs v
        this_word_logprob = torch.gather(this_word_logprob, 2, selected_words.unsqueeze(-1)) # N bs 1
        self.log_probs = list(
            torch.gather(o, 1, selected_beam.unsqueeze(-1).expand(self.b_s, self.beam_size, 1)) for o in self.log_probs) # 之前时刻加入的分数将作为下一时刻的beam
        self.log_probs.append(this_word_logprob) # 加入当前时刻选择词的分数
        self.selected_words = selected_words.view(-1, 1) # bs 1

        return visual, outputs
    
    def step(self, t, model, visual, cap, cap_mask, outputs):
        '''
        cap: N bs v
        word_logprob: bs N v
        outputs: from t=1
        '''
        # alpha = 0.9**(t)
        alpha = 1
        temp = 1
        k = 7
        
        if t == 0:
            predictions, _, _ = model(visual, cap[:, 0, :], cap_mask[:, 0, :])
            predictions0 = nn.Softmax(dim=1)(predictions[0][:, 0, :])
            predictions1 = nn.Softmax(dim=1)(predictions[1][:, 0, :])
            predictions2 = nn.Softmax(dim=1)(predictions[2][:, 0, :])
            
            score0 = torch.max(predictions0, dim=-1)[0]
            score1 = torch.max(predictions1, dim=-1)[0]
            score2 = torch.max(predictions2, dim=-1)[0]
                
            idx = torch.argmax(torch.Tensor([score0, score1, score2]))
            
            res = F.softmax(predictions[idx][:, 0, :], dim=-1)
            # res = F.softmax(model(visual, cap[:, 0, :], cap_mask[:, 0, :])[:, 0, :], dim=-1)
            # idx = torch.sort(res, descending=True)[1]
            # idxh = idx[:, :k][0]
            # idxl = idx[:, k:][0]
            # p = torch.sum(res[:, idxh], dim=-1)
            # res[:, idxl] = 0
            # res = res/p
            
            # predictions = alpha * F.log_softmax(model(visual, cap[:, 0, :], cap_mask[:, 0, :])[:, 0, :]/temp, dim=-1) # 1,8
            predictions_all = alpha * torch.log(res)
            predictions_all = predictions_all[:, None, :]
            
        else:
            predictions_all = []
            t0 = cap[:, :, 0].unsqueeze(-1) # N bs 1
            t1 = torch.cat(outputs, -1) # N bs t (t_max=5)
            t2 = cap[:, :, 1 + t1.size(-1):] # N bs 5-t

            cap = torch.cat((t0, t1, t2), dim=2)
            cap_mask = (cap >= 6).to(torch.bool)

            for bs in range(self.beam_size):
                predictions, _, _ = model(visual, cap[:, bs, :], cap_mask[:, bs, :])
                predictions0 = nn.Softmax(dim=1)(predictions[0][:, t, :])
                predictions1 = nn.Softmax(dim=1)(predictions[1][:, t, :])
                predictions2 = nn.Softmax(dim=1)(predictions[2][:, t, :])
                
                score0 = torch.max(predictions0, dim=-1)[0]
                score1 = torch.max(predictions1, dim=-1)[0]
                score2 = torch.max(predictions2, dim=-1)[0]
                    
                idx = torch.argmax(torch.Tensor([score0, score1, score2]))
                
                res = F.softmax(predictions[idx][:, t, :], dim=-1)
                
                # res = F.softmax(model(visual, cap[:, bs, :], cap_mask[:, bs, :])[:, t, :], dim=-1)
                # idx = torch.sort(res, descending=True)[1]
                # idxh = idx[:, :k][0]
                # idxl = idx[:, k:][0]
                # p = torch.sum(res[:, idxh], dim=-1)
                # res[:, idxl] = 0
                # res = res/p
                
                pred = alpha * torch.log(res)
                # pred = alpha * F.log_softmax(model(visual, cap[:, bs, :], cap_mask[:, bs, :])[:, t, :]/temp, dim=-1)
                predictions_all.append(pred)
                
                
            predictions_all = torch.stack(predictions_all, dim=1).transpose(1, 0)
            
        return predictions_all