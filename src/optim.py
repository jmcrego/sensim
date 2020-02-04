import torch
import logging
import sys
from torch import nn
from torch.autograd import Variable

class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        logging.debug('built NoamOpt')
        
    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def state_dict(self):
        return {
            "_step": self._step,
            "warmup": self.warmup,
            "factor": self.factor,
            "model_size": self.model_size,
            "_rate": self._rate,
            "optimizer": self.optimizer.state_dict()
        }

    def load_state_dict(self, state_dict):
        for key, value in state_dict.items():
            if key == "optimizer":
                self.optimizer.load_state_dict(state_dict["optimizer"])
            else:
                setattr(self, key, value)


##################################################################
### Criterions ###################################################
##################################################################

class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size #vocab size
        #self.true_dist = None
        logging.info('built criterion (label smoothing)')
        
    def forward(self, x, target): 
        #x is [batch_size*max_len, vocab] 
        #target is [batch_size*max_len]
        assert x.size(1) == self.size
        true_dist = x.data.clone() #[batch_size*max_len, vocab]
        true_dist.fill_(self.smoothing / (self.size - 2)) #true_dist is filled with value=smoothing/(size-2)
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence) #moves value=confidence to tensor true_dist and indiceds target_data dim=1
        true_dist[:, self.padding_idx] = 0 #prob mass on padding_idx is 0
        mask = torch.nonzero(target.data == self.padding_idx) # device=x.device ???
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        #self.true_dist = true_dist
        return self.criterion(x, true_dist) #total loss of this batch (not normalized)


class CosineSIM(nn.Module):
    def __init__(self, margin=0.0):
        super(CosineSIM, self).__init__()
        self.criterion = nn.CosineEmbeddingLoss(margin=margin, size_average=None, reduce=None, reduction='sum')
        logging.info('built criterion (cosine)')
        
    def forward(self, s1, s2, target):
        return self.criterion(s1, s2, target) #total loss of this batch (not normalized)


class AlignSIM(nn.Module):
    def __init__(self):
        super(AlignSIM, self).__init__()
        logging.info('built criterion (align)')
        
    def forward(self, aggr, y, mask_t):
        sign = torch.ones(aggr.size(), device=y.device) * y.unsqueeze(-1) #[b,lt] (by default ones builds on CPU)
        error = torch.log(1.0 + torch.exp(aggr * sign)) #equation (3) error of each tgt word
        print('error',error[1])
        sum_error = torch.sum(error * mask_t, dim=1) #error of each sentence in batch
        print('sum_error',sum_error[1])
        return torch.sum(sum_error) #total loss of this batch (not normalized)



##################################################################
### Compute losses ###############################################
##################################################################

class ComputeLossMLM:
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, h, y): 
        x_hat = self.generator(h) # project x softmax #[bs,sl,V]
        x_hat = x_hat.contiguous().view(-1, x_hat.size(-1)) #[bs*sl,V]
        y = y.contiguous().view(-1) #[bs*sl]
        loss = self.criterion(x_hat, y) 
        return loss #not normalized


class ComputeLossSIM:
    def __init__(self, criterion, pooling, R, opt=None):
        self.criterion = criterion
        self.pooling = pooling
        self.opt = opt
        self.R = R

    def __call__(self, hs, ht, slen, tlen, y, mask_s, mask_t): 
        #hs [bs, sl, es] embeddings of source words after encoder (<cls> <bos> s1 s2 ... sI <eos> <pad> ...)
        #ht [bs, tl, es] embeddings of target words after encoder (<cls> <bos> t1 t2 ... tJ <eos> <pad> ...)
        #slen [bs] length of source sentences (I) in batch
        #tlen [bs] length of target sentences (J) in batch
        #y [bs] parallel(1.0)/non_parallel(-1.0) value of each sentence pair
        #mask_s [bs,sl]
        #mask_t [bs,tl]
        #mask_st [bs,sl,tl]
        mask_s = mask_s.unsqueeze(-1).type(torch.float64)
        mask_t = mask_t.unsqueeze(-1).type(torch.float64)

        if self.pooling == 'max':
            s, _ = torch.max(hs*mask_s + (1.0-mask_s)*-999.9, dim=1) #-999.9 should be -Inf but it produces an nan when multiplied by 0.0
            t, _ = torch.max(ht*mask_t + (1.0-mask_t)*-999.9, dim=1) 
            loss = self.criterion(s, t, y)

        elif self.pooling == 'mean':
            s = torch.sum(hs * mask_s, dim=1) / torch.sum(mask_s, dim=1)
            t = torch.sum(ht * mask_t, dim=1) / torch.sum(mask_t, dim=1)
            loss = self.criterion(s, t, y)

        elif self.pooling == 'cls':
            s = hs[:, 0, :] # take embedding of first token <cls>
            t = ht[:, 0, :] # take embedding of first token <cls>
            loss = self.criterion(s, t, y)

        elif self.pooling == 'align':
            S_st = torch.bmm(hs, torch.transpose(ht, 2, 1)) #[bs, sl, es] x [bs, es, tl] = [bs, sl, tl]            
            aggr_t = self.aggr(S_st,mask_s) #equation (2) #for each tgt word, consider the aggregated matching scores over the source sentence words
            loss = self.criterion(aggr_t,y,mask_t.squeeze())
            print('loss',loss)
            sys.exit()

        else:
            logging.error('bad pooling method {}'.format(self.pooling))
            sys.exit()

        return loss #not normalized

    def aggr(self,S_st,mask_s): #foreach tgt word finds the aggregation over all src words
        print('S_st',S_st[1])
        #print('mask_s',mask_s[0])
        S_st_limited = torch.min(S_st, (torch.ones(S_st.size(), device=S_st.device)*9.9))
        print('S_st_limited',S_st_limited[1])
        exp_rS = torch.exp(S_st_limited * self.R)  ### attention!!! exp(large number) = nan
        print('exp_rS',exp_rS[1])
        sum_exp_rS = torch.sum(exp_rS * mask_s,dim=1) #sum over all source words (source words nor used are masked)
        print('sum_exp_rS',sum_exp_rS[1])
        log_sum_exp_rS_div_R = torch.log(sum_exp_rS) / self.R
        print('log_sum_exp_rS_div_R',log_sum_exp_rS_div_R[1])
        log_sum_exp_rS_div_R_limited, _ = torch.max(log_sum_exp_rS_div_R, (torch.ones(log_sum_exp_rS_div_R.size(), device=S_st.device)*-99.9))
        print('log_sum_exp_rS_div_R_limited',log_sum_exp_rS_div_R_limited[1])
        return log_sum_exp_rS_div_R_limited




