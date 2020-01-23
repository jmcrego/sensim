import numpy as np
import torch
import logging
import time
import random
import sys
import glob
import os
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from src.dataset import Vocab, DataSet, OpenNMTTokenizer
from src.model import make_model, ComputeLossMsk, ComputeLossSim
from src.optim import NoamOpt, LabelSmoothing, CosineSim

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
        self.msk_step = opts.train['msk_step']
        self.sim_step = opts.train['sim_step']
        self.ali_step = opts.train['ali_step']
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
        self.optimizer = NoamOpt(d_model, factor, warmup_steps, torch.optim.Adam(self.model.parameters(), lr=lrate, betas=(beta1, beta2), eps=eps))
        self.criterion_msk = LabelSmoothing(size=V, padding_idx=self.vocab.idx_pad, smoothing=label_smoothing)
        self.criterion_sim = CosineSim(tok1_idx=self.vocab.idx_cls, tok2_idx=self.vocab.idx_sep)
        if self.cuda:
            self.model.cuda()
            self.criterion.cuda()
        self.load_checkpoint() #loads if exists
        self.loss_msk = ComputeLossMsk(self.model.generator, self.criterion_msk, self.optimizer)
        self.loss_sim = ComputeLossSim(self.criterion_sim, self.optimizer)
        token = OpenNMTTokenizer(**opts.cfg['token'])


        logging.info('Read Train data')
        sim_uneven = opts.train['sim_step']['uneven']
        ali_uneven = opts.train['ali_step']['uneven']
        self.data_train = DataSet(opts.train['train'],token,self.vocab,opts.train['batch_size'][0],max_length=opts.train['max_length'],sim_uneven=sim_uneven,ali_uneven=ali_uneven,allow_shuffle=True,single_epoch=False)

        if 'valid' in opts.train:
            logging.info('read Valid data')
            self.data_valid = DataSet(opts.train['valid'],token,self.vocab,opts.train['batch_size'][0],max_length=opts.train['max_length'],allow_shuffle=True,single_epoch=True)
        else: 
            self.data_valid = None


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


    def __call__(self):
        logging.info('Start train n_steps_so_far={}'.format(self.n_steps_so_far))
        n_words_so_far = 0
        sum_loss_so_far = 0.0
        start = time.time()
        for step, batch, batch_tgt, batch_isparallel in self.data_train:
            #step is 'msk', 'sim', 'ali'
            #batch [batch_size, max_len] contain word_idx
            #batch_tgt [batch_size, max_len] contain word_idx of tgt words if step == 'ali'
            #batch_parallel [batch_size] contains +1.0 (parallel) or -1.0 (not parallel) sentences
            self.model.train()
            ###
            ### run step
            ###
            if step == 'msk':
                x, x_mask, y_mask, n_topredict = self.msk_batch_cuda(batch)
                #x contains the true words in batch after some masked (<msk>, random, same)
                #x_mask contains true for padded words, false for not padded words in batch
                #y_mask contains the source true words to predict of masked words, <pad> otherwise
                if n_topredict == 0: #nothing to predict
                    logging.info('batch with nothing to predict')
                    continue
                h = self.model.forward(x,x_mask) 
                loss = self.loss_msk(h, y_mask, n_topredict)
            elif step == 'sim':
                x, x_mask, y = self.sim_batch_cuda(batch,batch_isparallel)
                #x contains the true words in batch
                #x_mask contains true for padded words, false for not padded words in batch
                #y contains +1.0 (parallel) or -1.0 (not parallel) for each sentence pair
                h = self.model.forward(x,x_mask)
                loss = self.loss_sim(x, h, y)
                n_topredict = 1
            elif step == 'ali':
                continue
            else:
                logging.info('bad step {}'.format(step))
                sys.exit()

            self.n_steps_so_far += 1
            n_words_so_far += n_topredict
            sum_loss_so_far += loss 
            ###
            ### report
            ###
            if self.report_every_steps > 0 and self.n_steps_so_far % self.report_every_steps == 0:
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
            if self.data_valid is not None and len(self.data_valid) and self.validation_every_steps > 0 and self.n_steps_so_far % self.validation_every_steps == 0:
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
            y = torch.from_numpy(batch)
            x, x_mask = self.mask_batch(batch, step) 
            if self.cuda:
                x = x.cuda()
                x_mask = x_mask.cuda()
                y = y.cuda()
            n_words_to_predict = x_mask.sum()
            if n_words_to_predict == 0:
                continue
            y_pred = self.model.forward(x,x_mask) 
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


    def msk_batch_cuda(self, batch):
        batch = np.asarray(batch)
        x = torch.as_tensor(batch) #[batch_size, max_len] the original words with padding
        x_mask = torch.as_tensor((batch != self.vocab.idx_pad)).unsqueeze(-2) #[batch_size, 1, max_len]
        y_mask = x.clone() #[batch_size, max_len]

        prob = self.msk_step['prob']
        same = self.msk_step['same']
        rand = self.msk_step['rand']
        mask = 1.0 - same - rand
        if mask <= 0.0:
            logging.error('p_mask={} l<= zero'.format(mask))
            sys.exit()

        n_topredict = 0
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                y_mask[i][j] = self.vocab.idx_pad ### all padded except those masked (to be predicted)
                if not self.vocab.is_reserved(x[i][j]):
                    r = random.random()     # float in range [0.0, 1,0)
                    if r < prob:            ### is masked
                        n_topredict += 1
                        y_mask[i][j] = x[i][j]   # use the original (true) word rather than <pad> 
                        q = random.random() # float in range [0.0, 1,0)
                        if q < same:        # same
                            pass
                        elif q < same+rand: # rand among all vocab words
                            x[i][j] = random.randint(7,len(self.vocab)-1) # int in range [7, |vocab|)
                        else:               # <msk>
                            x[i][j] = self.vocab.idx_msk

        if self.cuda:
            x = x.cuda()
            x_mask = x_mask.cuda()
            y_mask = y_mask.cuda()

        return x, x_mask, y_mask, n_topredict


    def sim_batch_cuda(self, batch, y):
        batch = np.asarray(batch)
        y = np.asarray(y)
        x = torch.from_numpy(batch) #[batch_size, max_len] the original words with padding
        x_mask = torch.as_tensor((batch != self.vocab.idx_pad)).unsqueeze(-2) #[batch_size, 1, max_len]
        y = torch.as_tensor(y)
        if self.cuda:
            x = x.cuda()
            x_mask = x_mask.cuda()
            y = y.cuda()

        return x, x_mask, y






