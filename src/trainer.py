import numpy as np
import torch
import logging
import time
import random
import sys
import glob
import os
from collections import defaultdict
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from src.dataset import Vocab, DataSet, OpenNMTTokenizer, batch
from src.model import make_model
from src.optim import NoamOpt, LabelSmoothing, CosineSIM, AlignSIM, ComputeLossMLM, ComputeLossSIM, CrossEntropy


def sequence_mask(lengths, mask_n_initials=0):
    bs = len(lengths)
    l = lengths.max()
    msk = np.cumsum(np.ones([bs,l],dtype=int), axis=1).T #[l,bs] (transpose to allow combine with lenghts)
    mask = (msk <= lengths-1) ### i use lenghts-1 because the last unpadded word is <eos> and i want it masked too
    if mask_n_initials:
        mask &= (msk > mask_n_initials)
    return mask.T #[bs,l]

def st_mask(slen,tlen,mask_n_initials=0):
    bs = len(slen)
    ls = slen.max()
    lt = tlen.max()
    msk = np.zeros([bs,ls,lt], dtype=bool)
    for b in range(bs):
        for s in range(mask_n_initials,slen[b]-1):  ### i use slen[b]-1 because the last unpadded word is <eos> and i want it masked too
            msk[b,s,mask_n_initials:tlen[b]-1] = True  ### i use tlen[b]-1 because the last unpadded word is <eos> and i want it masked too
    #print('st_mask',msk)
    return msk

class stats():
    def __init__(self):
        self.n_preds = 0
        self.sum_loss = 0.0
        self.start = time.time()

    def add_batch(self,batch_loss,n_predicted):
        self.sum_loss += batch_loss
        self.n_preds += n_predicted

    def report(self,n_steps,step,trn_val_tst):
        logging.info("{} step: {} ({}) Loss: {:.4f} Predictions/sec: {:.1f}".format(trn_val_tst, n_steps, step, self.sum_loss/self.n_preds, self.n_preds/(time.time()-self.start)))
        self.n_preds = 0
        self.sum_loss = 0.0
        self.start = time.time()

class Trainer():

    def __init__(self, opts):
        self.dir = opts.dir
        self.report_every_steps = opts.train['report_every_steps']
        self.validation_every_steps = opts.train['validation_every_steps']
        self.checkpoint_every_steps = opts.train['checkpoint_every_steps']
        self.train_steps = opts.train['train_steps']
        self.vocab = Vocab(opts.cfg['vocab'])
        self.cuda = opts.cfg['cuda']
        self.n_steps_so_far = 0
        self.average_last_n = opts.train['average_last_n']
        self.steps = opts.train['steps']
        V = len(self.vocab)
        N = opts.cfg['num_layers']
        d_model = opts.cfg['hidden_size']
        d_ff = opts.cfg['feedforward_size']
        h = opts.cfg['num_heads']
        dropout = opts.cfg['dropout']
        factor = opts.cfg['factor']
        label_smoothing = opts.cfg['label_smoothing']
        warmup_steps = opts.cfg['warmup_steps']
        lrate = opts.cfg['learning_rate']
        beta1 = opts.cfg['beta1']
        beta2 = opts.cfg['beta2']
        eps = opts.cfg['eps']

        self.model = make_model(V, N=N, d_model=d_model, d_ff=d_ff, h=h, dropout=dropout)
        if self.cuda:
            self.model.cuda()

        self.optimizer = NoamOpt(d_model, factor, warmup_steps, torch.optim.Adam(self.model.parameters(), lr=lrate, betas=(beta1, beta2), eps=eps))

        if self.steps['sim']['run']:
            if self.steps['sim']['pooling'] == 'align':
                self.criterion = AlignSIM()
            else:
                self.criterion = CosineSIM()
        else:
            #self.criterion = CrossEntropy(padding_idx=self.vocab.idx_pad)
            self.criterion = LabelSmoothing(size=V, padding_idx=self.vocab.idx_pad, smoothing=label_smoothing)

        if self.cuda:
            self.criterion.cuda()

        self.load_checkpoint() #loads if exists

        if self.steps['sim']['run']:
            self.computeloss = ComputeLossSIM(self.criterion, self.steps['sim']['pooling'], self.steps['sim']['R'], self.optimizer)
        else:
            self.computeloss = ComputeLossMLM(self.model.generator, self.criterion, self.optimizer)
        token = OpenNMTTokenizer(**opts.cfg['token'])

        logging.info('read Train data')
        self.data_train = DataSet(self.steps,opts.train['train'],token,self.vocab,opts.train['batch_size'][0],max_length=opts.train['max_length'],swap_bitext=opts.train['swap_bitext'],allow_shuffle=True,valid_test=False)

        if 'valid' in opts.train:
            logging.info('read Valid data')
            self.data_valid = DataSet(self.steps,opts.train['valid'],token,self.vocab,opts.train['batch_size'][1],max_length=opts.train['max_length'],swap_bitext=opts.train['swap_bitext'],allow_shuffle=True,valid_test=True)
        else: 
            self.data_valid = None


    def __call__(self):
        logging.info('Start train n_steps_so_far={}'.format(self.n_steps_so_far))
        ts = stats()
        for batch in self.data_train:
            self.model.train()
            ###
            ### run step
            ###
            if not self.steps['sim']['run']: ### pre-training (MLM)
                step = 'mlm'
                x, x_mask, y_mask = self.mlm_batch_cuda(batch)
                #x contains the true words in batch after masking some of them (<msk>, random, same)
                #x_mask contains true for words to be predicted (masked), false otherwise
                #y_mask contains the original words of those cells to be predicted, <pad> otherwise
                n_predictions = torch.sum((y_mask != self.vocab.idx_pad)).data
                if n_predictions == 0: 
                    logging.info('batch with nothing to predict')
                    continue
                h = self.model.forward(x,x_mask)
                batch_loss = self.computeloss(h, y_mask)
                loss = batch_loss / n_predictions
                self.optimizer.zero_grad() 
                loss.backward()
                self.optimizer.step()
            else: ### fine-tunning (SIM)
                step = 'sim'
                x1, x2, l1, l2, x1_mask, x2_mask, y, mask_s, mask_t = self.sim_batch_cuda(batch) 
                #x1 contains the true words in batch_src
                #x2 contains the true words in batch_tgt
                #l1 length of sentences in batch
                #l2 length of sentences in batch
                #x1_mask contains true for padded words, false for not padded words in x1
                #x2_mask contains true for padded words, false for not padded words in x2
                #y contains +1.0 (parallel) or -1.0 (not parallel) for each sentence pair
                #mask_s is the source sequence_length mask true/false depending on padded source words
                #mask_t is the target sequence_length mask true/false depending on padded target words
                n_predictions = x1.size(0)
                h1 = self.model.forward(x1,x1_mask)
                h2 = self.model.forward(x2,x2_mask)
                #print('h1',h1.size())
                #print(h1)
                batch_loss = self.computeloss(h1, h2, l1, l2, y, mask_s, mask_t)
                loss = batch_loss / n_predictions
                self.optimizer.zero_grad() 
                loss.backward()
                self.optimizer.step()

            self.n_steps_so_far += 1
            ts.add_batch(batch_loss,n_predictions)
            ###
            ### report
            ###
            if self.report_every_steps > 0 and self.n_steps_so_far % self.report_every_steps == 0:
                ts.report(self.n_steps_so_far,step,'Train')
            ###
            ### saved
            ###
            if self.checkpoint_every_steps > 0 and self.n_steps_so_far % self.checkpoint_every_steps == 0:
                self.save_checkpoint()
            ###
            ### validation
            ###
            if self.data_valid is not None and self.validation_every_steps > 0 and self.n_steps_so_far % self.validation_every_steps == 0:
                self.validation()
            ###
            ### stop training
            ###
            if self.n_steps_so_far >= self.train_steps:
                break

        self.save_checkpoint()
        logging.info('End train')


    def validation(self):
        logging.info('Start validation')
        ds = stats()
        self.model.eval() ### avoids dropout
        for batch in self.data_valid:
            if not self.steps['sim']['run']: ### pre-training (MLM)
                step = 'mlm'
                x, x_mask, y_mask = self.mlm_batch_cuda(batch)
                n_predictions = torch.sum((y_mask != self.vocab.idx_pad)).data
                if n_predictions == 0: #nothing to predict
                    logging.info('batch with nothing to predict')
                    continue
                h = self.model.forward(x,x_mask)
                batch_loss = self.computeloss(h, y_mask)
#                del x, x_mask, y_mask
            else: ### fine-tunning (SIM)
                step = 'sim'
                x1, x2, l1, l2, x1_mask, x2_mask, y, mask_s, mask_t = self.sim_batch_cuda(batch) 
                n_predictions = x1.size(0)
                h1 = self.model.forward(x1,x1_mask)
                h2 = self.model.forward(x2,x2_mask)
                batch_loss = self.computeloss(h1, h2, l1, l2, y, mask_s, mask_t)
#                del x1, x2, l1, l2, x1_mask, x2_mask, y, mask_s, mask_t
#            torch.cuda.empty_cache()
            ds.add_batch(batch_loss,n_predictions)
        ds.report(self.n_steps_so_far,step,'Valid')
        logging.info('End validation')


    def mlm_batch_cuda(self, batch):
        batch = np.array(batch.idx_src)
        x = torch.from_numpy(batch) #[batch_size, max_len] contains the original words. some will be masked
        x_mask = torch.as_tensor((batch != self.vocab.idx_pad)).unsqueeze(-2) #[batch_size, 1, max_len]. Contains true for words to be predicted (masked), false otherwise
        y_mask = torch.ones_like(x, dtype=torch.int64) #[batch_size, max_len]. will contain the original value of masked words in x. <pad> for the rest

        p_mask = self.steps['mlm']['p_mask']
        r_same = self.steps['mlm']['r_same']
        r_rand = self.steps['mlm']['r_rand']
        r_mask = 1.0 - r_same - r_rand

        if r_mask <= 0.0:
            logging.error('mask={} l<= zero'.format(mask))
            sys.exit()

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                y_mask[i,j] = self.vocab.idx_pad ### all padded except those masked (to be predicted)
                if not self.vocab.is_reserved(x[i,j]):
                    r = random.random()     # float in range [0.0, 1,0)
                    if r < p_mask:          ### is masked
                        y_mask[i,j] = x[i,j]   # use the original (true) word rather than <pad> 
                        q = random.random() # float in range [0.0, 1,0)
                        if q < r_same:        # same
                            pass
                        elif q < r_same+r_rand: # rand among all vocab words
                            x[i,j] = random.randint(7,len(self.vocab)-1) # int in range [7, |vocab|)
                        else:               # <msk>
                            x[i,j] = self.vocab.idx_msk

        if self.cuda:
            x = x.cuda()
            x_mask = x_mask.cuda()
            y_mask = y_mask.cuda()

        return x, x_mask, y_mask


    def sim_batch_cuda(self, batch):
        batch_src = np.array(batch.idx_src)
        batch_tgt = np.array(batch.idx_tgt)
        batch_src_len = np.array(batch.lsrc)
        batch_tgt_len = np.array(batch.ltgt)
        y = np.array(batch.isParallel)

        x1 = torch.from_numpy(batch_src) #[batch_size, max_len] the original words with padding
        x1_mask = torch.as_tensor((batch_src != self.vocab.idx_pad)).unsqueeze(-2) #[batch_size, 1, max_len]
        l1 = torch.from_numpy(batch_src_len) #[bs]

        x2 = torch.from_numpy(batch_tgt) #[batch_size, max_len] the original words with padding
        x2_mask = torch.as_tensor((batch_tgt != self.vocab.idx_pad)).unsqueeze(-2) #[batch_size, 1, max_len]
        l2 = torch.from_numpy(batch_tgt_len) #[bs]

        mask_s = torch.from_numpy(sequence_mask(batch_src_len,mask_n_initials=2)) 
        mask_t = torch.from_numpy(sequence_mask(batch_tgt_len,mask_n_initials=2)) 

        y = torch.as_tensor(y)

        if self.cuda:
            x1 = x1.cuda()
            x1_mask = x1_mask.cuda()
            l1 = l1.cuda()
            x2 = x2.cuda()
            x2_mask = x2_mask.cuda()
            l2 = l2.cuda()
            y = y.cuda()
            mask_s = mask_s.cuda()
            mask_t = mask_t.cuda()

        return x1, x2, l1, l2, x1_mask, x2_mask, y, mask_s, mask_t


    def load_checkpoint(self):
        files = sorted(glob.glob(self.dir + '/checkpoint.???????.pth')) 
        if len(files):
            file = files[-1] ### last is the newest
            checkpoint = torch.load(file)
            self.n_steps_so_far = checkpoint['n_steps_so_far']
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.model.load_state_dict(checkpoint['model'])
            logging.info('loaded checkpoint {}'.format(file))
        else:
            logging.info('no checkpoint available')

            
    def save_checkpoint(self):
        file = '{}/checkpoint.{:07d}.pth'.format(self.dir,self.n_steps_so_far)
        state = {
            'n_steps_so_far': self.n_steps_so_far,
            'optimizer': self.optimizer.state_dict(),
            'model': self.model.state_dict()
        }
        torch.save(state, file)
        logging.info('saved checkpoint {}'.format(file))
        files = sorted(glob.glob(self.dir + '/checkpoint.???????.pth')) 
        while len(files) > self.average_last_n:
            f = files.pop(0)
            os.remove(f) ### first is the oldest
            logging.debug('removed checkpoint {}'.format(f))


    def average(self):
        files = sorted(glob.glob(self.dir + '/checkpoint.???????.pth')) 
        if len(files) == 0:
            logging.info('no checkpoint available')
            return
        #read models
        models = []
        for file in files:
            m = self.model.clone()
            checkpoint = torch.load(file)
            m.load_state_dict(checkpoint['model'])
            models.append(m)
            logging.info('read {}'.format(file))
        #create mout 
        mout = self.model.clone()
        for ps in zip(*[m.params() for m in [mout] + models]):
            p[0].copy_(torch.sum(*ps[1:]) / len(ps[1:]))
        fout = self.dir + '/checkpoint.averaged.pth'
        state = {
            'n_steps_so_far': mout.n_steps_so_far,
            'optimizer': mout.optimizer.state_dict(),
            'model': mout.model.state_dict()
        }
        #save mout into fout
        torch.save(state, fout)
        logging.info('averaged {} models into {}'.format(len(files), fout))





