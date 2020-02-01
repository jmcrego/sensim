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
        
#def get_std_opt(model):
#    return NoamOpt(model.src_embed[0].d_model, 2, 4000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

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
        self.size = size
        self.true_dist = None
        logging.debug('built criterion (label smoothing)')
        
    def forward(self, x, target): #x is [batch_size*max_len, embedding_size] target is [batch_size*max_len, 1]
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


class CosineSIM(nn.Module):
    def __init__(self, margin=0.0):
        super(CosineSIM, self).__init__()
        self.criterion = nn.CosineEmbeddingLoss(margin=margin, size_average=None, reduce=None, reduction='mean')
        logging.debug('built criterion (cosine)')
        
    def forward(self, s1, s2, target):
        return self.criterion(s1, s2, target)


class AlignSIM(nn.Module):
    def __init__(self):
        super(AlignSIM, self).__init__()
        #self.criterion = nn.MSELoss(size_average=None, reduce=None, reduction='mean') 
        logging.debug('built criterion (align)')
        
    def forward(self, aggr, y, mask_t):
        sign = torch.ones(aggr.size()) * y.unsqueeze(-1) 
        error = torch.log(1.0 + torch.exp(aggr * sign)) #equation (3) error of each tgt word
        sum_error = torch.sum(error * mask_t, dim=1) #error of each sentence
        loss = torch.mean(sum_error) 
        return loss

##################################################################
### Compute losses ###############################################
##################################################################

class ComputeLossMLM:
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, h, y, n_topredict): 
#        if self.opt is not None:
#            self.opt.optimizer.zero_grad() #resets parameters gradients (x.grad)
        x_hat = self.generator(h) # project x softmax 
        #x_hat [batch_size, max_len, |vocab|]
        #y     [batch_size, max_len]
        x_hat = x_hat.contiguous().view(-1, x_hat.size(-1))
        y = y.contiguous().view(-1)
        #x_hat [batch_size*max_len, |vocab|]
        #y     [batch_size*max_len]

        #n_ok = ((y == torch.argmax(x_hat, dim=1)) * (y != self.criterion.padding_idx)).sum()
        #logging.debug('batch {}/{} Acc={:.2f}'.format(n_ok,n_topredict,100.0*n_ok/n_topredict))

        loss = self.criterion(x_hat, y) / n_topredict #(normalised per token predicted)_
#        loss.backward()
#        if self.opt is not None:
#            self.opt.step() #performs a parameter update based on the current gradient
        return loss * n_topredict


class ComputeLossSIM:
    def __init__(self, criterion, pooling, opt=None):
        self.criterion = criterion
        self.pooling = pooling
        self.opt = opt
        self.R = 1.0

    def __call__(self, hs, ht, slen, tlen, y, mask_s, mask_t, mask_st): 
        #hs [bs, sl, es] embeddings of source words after encoder (<cls> <bos> s1 s2 ... sI <eos> <pad> ...)
        #ht [bs, tl, es] embeddings of target words after encoder (<cls> <bos> t1 t2 ... tJ <eos> <pad> ...)
        #slen [bs] length of source sentences (I) in batch
        #tlen [bs] length of target sentences (J) in batch
        #y  [bs] parallel(1.0)/non_parallel(-1.0) value of each sentence pair
        #mask_s [bs,ls]
        #mask_t [bs,tl]
        #mask_st [bs,sl,tl]
        #print('mask_s',mask_s.size())
        #print(mask_s)
        #print('mask_t',mask_t.size())
        #print(mask_t)
        #print('mask_st',mask_st.size())
        #print(mask_st)
        #print('hs',hs.size())
        #print(hs)
        #print('ht',ht.size())
        #print(ht)
        #print('slen',slen.size())
        #print('tlen',tlen.size())
        #print('y',y.size())
        mask_s = mask_s.unsqueeze(-1).type(torch.float64)
        mask_t = mask_t.unsqueeze(-1).type(torch.float64)
        mask_st = mask_st.type(torch.float64)

#        if self.opt is not None:
#            self.opt.optimizer.zero_grad() #resets parameters gradients (x.grad)

        if self.pooling == 'max':
            s, _ = torch.max(hs*mask_s + (1.0-mask_s)*-float('Inf'), dim=1)
            t, _ = torch.max(ht*mask_t + (1.0-mask_t)*-float('Inf'), dim=1)
            loss = self.criterion(s, t, y)

        elif self.pooling == 'mean':
            s = torch.sum(hs * mask_s, dim=1) / torch.sum(mask_s, dim=1)
            t = torch.sum(ht * mask_t, dim=1) / torch.sum(mask_t, dim=1)
            loss = self.criterion(s, t, y)
            print('loss (mean over batch examples)',loss)

        elif self.pooling == 'cls':
            s = hs[:, 0, :] # take embedding of first token <cls>
            t = ht[:, 0, :] # take embedding of first token <cls>
            loss = self.criterion(s, t, y)
            print('loss (mean over batch examples)',loss)

        elif self.pooling == 'align':
            S_st = torch.bmm(hs, torch.transpose(ht, 2, 1)) #[bs, sl, es] x [bs, es, tl] = [bs, sl, tl]
            aggr_t = self.aggr(S_st,mask_s) #equation (2) #for each tgt word, consider the aggregated matching scores over the source sentence words
            loss = self.criterion(aggr_t,y,mask_t.squeeze())
            print('loss (mean over batch examples)',loss)

        else:
            logging.error('bad pooling method {}'.format(self.pooling))
            sys.exit()

#        loss.backward() #computes gradients dloss/dx for parameters x (only x's with requires_grad=True) and accumulates them in x.grad (x.grad += dloss/dx)
#        if self.opt is not None:
#            self.opt.step() #For each parameter x, it performs the parameter update. (x += -lr * x.grad)
        return loss

    def aggr(self,S_st,mask_s): #foreach tgt word finds the aggregation over all src words
        #print('S_st',S_st.size()) #[bs, ls, lt]
        #print('mask_s',mask_s.size())
        #print('mask_st',mask_st.size()) #[bs, ls, lt] contains zero those cells to be padded
        #print(mask_st)
        exp_rS = torch.exp(S_st * self.R)
        #print('exp_rS',exp_rS.size())
        #print(exp_rS)
        sum_exp_rS = torch.sum(exp_rS * mask_s,dim=1) #sum over all source words (source words nor used are masked)
        #print('sum_exp_rS (sum over src words)',sum_exp_rS.size()) #[bs,lt]
        #print(sum_exp_rS)
        log_sum_exp_rS_div_R = torch.log(sum_exp_rS) / self.R
        #print('log_sum_exp_rS_div_R',log_sum_exp_rS_div_R.size()) #[bs,lt]
        #print(log_sum_exp_rS_div_R)
        return log_sum_exp_rS_div_R

'''
def sequence_mask(lengths, mask_n_initials=0):
    maxlen = lengths.max()
    msk = torch.ones([len(lengths), maxlen]).cumsum(dim=1, dtype=torch.int32).t()
    #for a 3x3 matrix: msk = [[1,2,3],[1,2,3],[1,2,3]]
    if mask_n_initials > 0:
        return ((msk <= lengths) & (msk > mask_n_initials)).t().type(torch.bool)
    return (msk <= lengths).t().type(torch.bool)


def st_mask(slen,tlen,mask_n_initials=0):
    assert len(slen)==len(tlen)
    bs = len(slen)
    ls = slen.max()
    lt = tlen.max()
    #print('matrix is [bs={} x [{},{}]]'.format(bs,ls,lt))
    #print('slen={}'.format(slen))
    #print('tlen={}'.format(tlen))
    msk = torch.zeros([bs,ls,lt], dtype=torch.bool)
    for b in range(bs):
        for s in range(mask_n_initials,slen[b]):
            msk[b,s,mask_n_initials:tlen[b]] = True
    #print(msk)
    return msk
'''




