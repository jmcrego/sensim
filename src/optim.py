import torch
import logging
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

        loss = self.criterion(x_hat, y) / n_topredict
        loss.backward()
        if self.opt is not None:
            self.opt.step() #performs a parameter update based on the current gradient
        return loss.data * n_topredict


class ComputeLossSim:
    def __init__(self, criterion, pooling, opt=None):
        self.criterion = criterion
        self.opt = opt
        self.pooling = pooling

    def __call__(self, h1, h2, mask1, mask2, y): 
        #h1 [batch_size, max_len, embedding_size] embeddings of source words after encoder
        #h2 [batch_size, max_len, embedding_size] embeddings of target words after encoder
        #y [batch_size, max_len] parallel(1.0)/non_parallel(-1.0) value of each sentence pair
        if self.opt is not None:
            self.opt.optimizer.zero_grad()

        if self.pooling == 'max':
            pass

        elif self.pooling == 'mean':
            pass

        elif self.pooling == 'cls':
            s1 = h1[:0:]
            s2 = h2[:0:]

        else: # 'align':
            pass

        loss = self.criterion(s1, s2, y)
        loss.backward()
        if self.opt is not None:
            self.opt.step() #performs a parameter update based on the current gradient

        return loss.data


