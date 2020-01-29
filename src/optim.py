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


class CosineSim(nn.Module):
    def __init__(self, margin=0.0):
        super(CosineSim, self).__init__()
        self.criterion = nn.CosineEmbeddingLoss(margin=margin, size_average=None, reduce=None, reduction='mean')
        logging.debug('built criterion (cosine)')
        
    def forward(self, s1, s2, target):
        return self.criterion(s1, s2, target)


class AlignSim(nn.Module):
    def __init__(self):
        super(AlignSim, self).__init__()
        self.criterion = nn.MSELoss(size_average=None, reduce=None, reduction='mean') 
        logging.debug('built criterion (align)')
        
    def forward(self, y_hat, target):
        return self.criterion(y_hat, target)

##################################################################
### Compute losses ###############################################
##################################################################

class ComputeLossMsk:
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, h, y, n_topredict): 
        if self.opt is not None:
            self.opt.optimizer.zero_grad()
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
        loss.backward()
        if self.opt is not None:
            self.opt.step() #performs a parameter update based on the current gradient
        return loss.data * n_topredict


class ComputeLossSim:
    def __init__(self, criterion, pooling, opt=None):
        self.criterion = criterion
        self.pooling = pooling
        self.opt = opt
        self.R = 1.0

    def __call__(self, hs, ht, mask_st, slen, tlen, y): 
        #hs [bs, sl, es] embeddings of source words after encoder (<cls> <bos> s1 s2 ... sI <eos> <pad> ...)
        #ht [bs, tl, es] embeddings of target words after encoder (<cls> <bos> t1 t2 ... tJ <eos> <pad> ...)
        #mask_st [bs, sl, tl] mask contain True for words to be considered and False otherwise
        #slen [bs] length of source sentences (I) in batch
        #tlen [bs] length of target sentences (J) in batch
        #y  [bs] parallel(1.0)/non_parallel(-1.0) value of each sentence pair
        print('hs',hs.size())
        print('ht',ht.size())
        print('slen',slen.size())
        print('tlen',tlen.size())
        print('y',y.size())
        if self.opt is not None:
            self.opt.optimizer.zero_grad()

        if self.pooling == 'max':
            mask_s = sequence_mask(slen,mask_n_initials=2).unsqueeze(-1)
            mask_t = sequence_mask(tlen,mask_n_initials=2).unsqueeze(-1)
            s, _ = torch.max(hs * mask_s + (1-mask_s) * torch.float('-Inf'), dim=1)
            t, _ = torch.max(ht * mask_t + (1-mask_t) * torch.float('-Inf'), dim=1)
            loss = self.criterion(s, t, y)

        elif self.pooling == 'mean':
            s = torch.mean(hs * sequence_mask(slen,mask_n_initials=2).unsqueeze(-1), dim=1)
            t = torch.mean(ht * sequence_mask(tlen,mask_n_initials=2).unsqueeze(-1), dim=1)
            loss = self.criterion(s, t, y)

        elif self.pooling == 'cls':
            s = hs[:, 0, :] # take embedding of first token <cls>
            t = ht[:, 0, :] # take embedding of first token <cls>
            loss = self.criterion(s, t, y)

        elif self.pooling == 'align':
            S = torch.bmm(hs, torch.transpose(ht, 2, 1)) #[bs, sl, es] x [bs, es, tl] = [bs, sl, tl]
            aggr = self.aggr(S,mask_st) #equation (2) #for each tgt word, consider the aggregated matching scores over the source sentence 
            error = torch.log(1 + torch.exp(aggr * sign)) #equation (3) error of each tgt word
            #sum_error = tf.map_fn(lambda (x, l): tf.reduce_sum(x[2:l-1]), (error_tgt, tl))
            loss = torch.sum(sum_error * mask_t) / torch.sum(mask_t)
            #loss = self.criterion(s, t, y)

        else:
            logging.error('bad pooling method {}'.format(self.pooling))
            sys.exit()

        loss.backward()
        if self.opt is not None:
            self.opt.step() #performs a parameter update based on the current gradient

        return loss.data

    def aggr(self,S_st,mask_st):
        print('S_st',S_st.size()) #[bs, ls, lt]
        print('mask_st',mask_st.size()) #[bs, ls, lt] contains zero those cells to be padded
        #print('l_s',l_s.size()) #[bs] #lengths of source sentences 
        #The aggregation operation summarizes the alignment scores for each target word
        exp_rS = torch.exp(S_st * self.R)
        print('exp_rS',exp_rS.size()) #[bs,ls,lt]

        #sum_exp_rS = tf.map_fn(lambda (x, l): tf.reduce_sum(x[2:l-1, :], 0), (exp_rS, l_s)) #[B,Ss] (do not sum over <bos> and <eos> [2:l-1])
        mask_zeros = torch.zeros(S.size(), requires_grad=False)
        sum_exp_rS = torch.sum(exp_rS * mask_st,dim=1) #sum over source words (dim=1)
        print('sum_exp_rS',sum_exp_rS.size()) #[bs,lt]
        log_sum_exp_rS = torch.log(sum_exp_rS) 
        print('log_sum_exp_rS',log_sum_exp_rS.size()) #[bs,lt]
        aggr = log_sum_exp_rS / self.R
        print('aggr',aggr.size()) #[bs,lt]
        sys.exit()
        return aggr


def mask_3Dbatch(batch, lengths, value, mask_n_initials=0):
    #input batch is [batch_size, seq_length, embedding_size]
    #i set to value those cells batch[b,l,k]:
    #l == 0 or 1 (if mask_initial) ===> thus masking <cls> and <bos>
    #l >= lengths[l]-1             ===> thus masking <eos> <pad> ...
    for b in range(batch.size(0)): #for each sentence batch[b]
        for l in range(batch.size(1)): #for each token batch[b,l]
            if l<mask_n_initials or l>=lengths[l]-1:
                batch[b,l] = [value] * batch.size(2) ### replace all embedding cells of this token by value
    return batch


def mask_3Dtensor(batch, lengths, value, mask_n_initials=0):
    batch_mask = batch.clone()
    #input batch is [batch_size, seq_length, embedding_size]
    #i set to value those cells batch[b,l,k]:
    #l == 0 or 1 (if mask_initial) ===> thus masking <cls> and <bos>
    #l >= lengths[l]-1             ===> thus masking <eos> <pad> ...
    for b in range(batch_mask.size(0)): #for each sentence batch[b]
        for l in range(batch_mask.size(1)): #for each token batch[b,l]
            if l<mask_n_initials or l>=lengths[l]-1:
                for c in range(batch_mask.size(2)): #for each cell batch[b,l,c]
                    batch_mask[b,l,c] = value ### replace all embedding cells of this token by value
    return batch_mask


def sequence_mask(lengths, mask_n_initials=0):
    maxlen = lengths.max()
    msk = torch.ones([len(lengths), maxlen]).cumsum(dim=1, dtype=torch.int32).t()
    #for a 3x3 matrix: msk = [[1,2,3],[1,2,3],[1,2,3]]
    if mask_n_initials > 0:
        return ((msk <= lengths) & (msk > mask_n_initials)) #.t().type(torch.bool)
    return (msk <= lengths) #.t().type(torch.bool)






