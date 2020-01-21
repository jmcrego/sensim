import numpy as np
import torch
import logging
import time
import random
import sys
import glob
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from src.dataset import Vocab, DataSet, OpenNMTTokenizer
from src.model import make_model

class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._step = 0
        self._rate = 0
        
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


class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
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


class ComputeLoss:
    def __init__(self, criterion, opt=None):
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        if self.opt is not None:
            self.opt.optimizer.zero_grad()
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step() #performs a parameter update based on the current gradient
        return loss.data * norm


class Trainer():

    def __init__(self, opts):
        self.dir = opts.dir
        self.report_every_steps = opts.train['report_every_steps']
        self.validation_every_steps = opts.train['validation_every_steps']
        self.checkpoint_every_steps = opts.train['checkpoint_every_steps']
        self.train_steps = opts.train['train_steps']
        self.vocab = Vocab(opts.mod['vocab'])
        self.cuda = opts.mod['cuda']
        self.n_steps_so_far = 0
        self.steps = []
        if 'mono_step' in opts.train and 'prob' in opts.train['mono_step'] and opts.train['mono_step']['prob'] > 0:
            self.steps.append('mono')
            self.mono_step = opts.train['mono_step']
        if 'para_step' in opts.train and 'prob' in opts.train['para_step'] and opts.train['para_step']['prob'] > 0:
            self.steps.append('para')
            self.para_step = opts.train['para_step']
        if 'tran_step' in opts.train and 'prob' in opts.train['tran_step'] and opts.train['tran_step']['prob'] > 0:
            self.steps.append('tran')
            self.tran_step = opts.train['tran_step']
        logging.debug('steps: {}'.format(self.steps))
        V = len(self.vocab)
        N = opts.mod['num_layers']
        d_model = opts.mod['hidden_size']
        d_ff = opts.mod['feedforward_size']
        h = opts.mod['num_heads']
        dropout = opts.mod['dropout']
        factor = opts.opt['factor']
        smoothing = opts.opt['smoothing']
        warmup_steps = opts.opt['warmup_steps']
        lrate = opts.opt['learning_rate']
        beta1 = opts.opt['beta1']
        beta2 = opts.opt['beta2']
        eps = opts.opt['eps']
        
        self.model = make_model(V, N=N, d_model=d_model, d_ff=d_ff, h=h, dropout=dropout)
        logging.debug('built model')

        self.optimizer = NoamOpt(d_model, factor, warmup_steps, torch.optim.Adam(self.model.parameters(), lr=lrate, betas=(beta1, beta2), eps=eps))
        logging.debug('built optimizer')

        self.criterion = LabelSmoothing(size=V, padding_idx=self.vocab.idx_pad, smoothing=smoothing)
        logging.debug('built criterion (label smoothing)')

        self.load_checkpoint() #loads if exists

        self.computeloss = ComputeLoss(self.criterion, self.optimizer)

        token_src = OpenNMTTokenizer(**opts.mod['tokenization']['src'])
        token_tgt = OpenNMTTokenizer(**opts.mod['tokenization']['tgt'])

        logging.info('Read Train data')
        self.data_train = DataSet(opts.train['batch_size'], is_valid=False)
        files_src = opts.train['train']['src']
        files_tgt = opts.train['train']['tgt']
        self.data_train.read(files_src,files_tgt,token_src,token_tgt,self.vocab,max_length=opts.train['max_length'],example=opts.mod['example_format'])

        logging.info('Read Valid data')
        batch_size = 4
        self.data_valid = DataSet(batch_size, is_valid=True)
        files_src = opts.train['valid']['src']
        files_tgt = opts.train['valid']['tgt']
        self.data_valid.read(files_src,files_tgt,token_src,token_tgt,self.vocab,max_length=0,example=opts.mod['example_format'])

    def load_checkpoint(self):
        files = sorted(glob.glob(self.dir + '/checkpoint.???????.pth')) 
        if len(files):
            file = files[-1]
            checkpoint = torch.load(file)
            self.cuda = checkpoint['cuda']
            if self.cuda:
                self.model.cuda()
                self.criterion.cuda()
            self.n_steps_so_far = checkpoint['n_steps_so_far']
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.model.load_state_dict(checkpoint['model'])
            logging.info('Loaded checkpoint {}'.format(file))
        else:
            logging.info('no checkpoint available')
            if self.cuda:
                self.model.cuda()
                self.criterion.cuda()

            

            
    def save_checkpoint(self):
        file = '{}/checkpoint.{:07d}.pth'.format(self.dir,self.n_steps_so_far)
        state = {
            'cuda': self.cuda,
            'n_steps_so_far': self.n_steps_so_far,
            'optimizer': self.optimizer.state_dict(),
            'model': self.model.state_dict()
        }
        torch.save(state, file)
        logging.info('Saved checkpoint {}'.format(file))


    def __call__(self):
        logging.info('Start train n_steps_so_far={}'.format(self.n_steps_so_far))
        self.data_train.build_batches()
        self.data_valid.build_batches()
        n_words_so_far = 0
        sum_loss_so_far = 0.0
        start = time.time()
        for batch, batch_len in self.data_train:
            ###
            ### run step
            ###
            self.model.train()
            batch = np.array(batch)
            step = self.steps[self.n_steps_so_far % len(self.steps)]
            y = torch.from_numpy(batch)
            x, x_mask = self.mask_batch(batch, step) 
            if self.cuda:
                x = x.cuda()
                x_mask = x_mask.cuda()
                y = y.cuda()
            n_words_to_predict = x_mask.sum()
            if n_words_to_predict == 0:
                logging.debug('no word to predict')
                continue
            y_pred = self.model.forward(x,x_mask.unsqueeze(-2)) #unsqueeze(-2) outputs [batch_size, 1, max_len]
            loss = self.computeloss(y_pred, y, n_words_to_predict)
            self.n_steps_so_far += 1
            n_words_so_far += n_words_to_predict
            sum_loss_so_far += loss 
            ###
            ### report
            ###
            if self.report_every_steps > 0 and self.n_steps_so_far % self.report_every_steps == 0:
                #logging.info('n_steps_so_far={}'.format(self.n_steps_so_far))
                logging.info("Train step: {} Loss: {:.4f} Tokens/sec: {:.1f}".format(self.n_steps_so_far, sum_loss_so_far / n_words_so_far, n_words_so_far / (time.time() - start))) 
                n_words_so_far = 0
                sum_loss_so_far = 0.0
                start = time.time()
            ###
            ### save
            ###
            if self.checkpoint_every_steps > 0 and self.n_steps_so_far % self.checkpoint_every_steps == 0:
                self.save_checkpoint()
            ###
            ### validation
            ###
            if self.validation_every_steps > 0 and self.n_steps_so_far % self.validation_every_steps == 0:
                self.validation()
            ###
            ### stop training
            ###
            if self.n_steps_so_far >= self.train_steps:
                break

        self.save_checkpoint()
        logging.info('End train')


    def validation(self):
        self.model.eval()
        n_steps_so_far = 0
        n_words_so_far = 0
        n_words_valid = 0
        sum_loss_valid = 0
        sum_loss_so_far = 0.0
        start = time.time()
        for batch, batch_len in self.data_valid:
            batch = np.array(batch)
            step = self.steps[n_steps_so_far % len(self.steps)]
            y = torch.from_numpy(batch)
            x, x_mask = self.mask_batch(batch, step) 
            if self.cuda:
                x = x.cuda()
                x_mask = x_mask.cuda()
                y = y.cuda()
            n_words_to_predict = x_mask.sum()
            if n_words_to_predict == 0:
                continue
            y_pred = self.model.forward(x,x_mask.unsqueeze(-2)) #unsqueeze(-2) outputs [batch_size, 1, max_len]
            loss = self.computeloss(y_pred, y, n_words_to_predict)
            n_steps_so_far += 1
            n_words_so_far += n_words_to_predict
            n_words_valid += n_words_to_predict
            sum_loss_so_far += loss 
            sum_loss_valid += loss
            ###
            ### report
            ###
            if self.report_every_steps > 0 and n_steps_so_far % self.report_every_steps == 0:
                logging.info("Valid step: {}/{} Loss: {:.4f} Tokens/sec: {:.1f}".format(n_steps_so_far, len(self.data_valid), sum_loss_so_far / n_words_so_far, n_words_so_far / (time.time() - start))) 
                n_words_so_far = 0
                sum_loss_so_far = 0.0
                start = time.time()
        logging.info('Valid Loss: {:.4f}'.format(sum_loss_valid / n_words_valid))


    def mask_batch(self, batch, step):
        prob = 0.0
        same = 0.0
        rnd = 0.0
        if step == 'mono':
            prob = self.mono_step['prob']
            same = self.mono_step['same']
            rnd = self.mono_step['rnd']
        elif step == 'para':
            prob = self.para_step['prob']
            same = self.para_step['same']
            rnd = self.para_step['rnd']
        elif step == 'tran':
            prob = self.tran_step['prob']
            same = self.tran_step['same']
            rnd = self.tran_step['rnd']
        else:
            logging.error('bad step name: {}'.format(step))
            sys.exit()
        mask = 1.0 - same - rnd
        if mask <= 0.0:
            logging.error('p_mask={} l<= zero'.format(mask))
            sys.exit()

        #x = torch.from_numpy(batch) #[batch_size, max_len]
        x = torch.as_tensor(batch) #[batch_size, max_len]
        x_mask = torch.zeros(x.shape, dtype=torch.bool, requires_grad=False) #True if it is masked (either: <msk> or same or rnd)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if not self.vocab.is_reserved(x[i][j]):
                    r = random.random()     # float in range [0.0, 1,0)
                    if r <= prob: #masked (either: <msk> or same or rnd)
                        x_mask[i][j] = True
                        q = random.random() # float in range [0.0, 1,0)
                        if q < same:        # same
                            pass
                        elif q < same+rnd:  # rnd among all vocab words
                            x[i][j] = random.randint(7,len(self.vocab)-1) # int in range [7, |vocab|)
                        else:               # <msk>
                            x[i][j] = self.vocab['<msk>']

        return x, x_mask

    def compute_loss(self, ):
        pass






