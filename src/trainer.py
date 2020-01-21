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
from src.optim import NoamOpt, LabelSmoothing, ComputeLoss

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
        N = opts.cfg['num_layers']
        d_model = opts.cfg['hidden_size']
        d_ff = opts.cfg['feedforward_size']
        h = opts.cfg['num_heads']
        dropout = opts.cfg['dropout']
        factor = opts.cfg['factor']
        smoothing = opts.cfg['smoothing']
        warmup_steps = opts.cfg['warmup_steps']
        lrate = opts.cfg['learning_rate']
        beta1 = opts.cfg['beta1']
        beta2 = opts.cfg['beta2']
        eps = opts.cfg['eps']
        
        self.model = make_model(V, N=N, d_model=d_model, d_ff=d_ff, h=h, dropout=dropout)
        self.optimizer = NoamOpt(d_model, factor, warmup_steps, torch.optim.Adam(self.model.parameters(), lr=lrate, betas=(beta1, beta2), eps=eps))
        self.criterion = LabelSmoothing(size=V, padding_idx=self.vocab.idx_pad, smoothing=smoothing)
        self.load_checkpoint() #loads if exists
        self.computeloss = ComputeLoss(self.criterion, self.optimizer)
        token = OpenNMTTokenizer(**opts.cfg['token'])

        logging.info('Read Train data')
        self.data_train = DataSet(opts.train['batch_size'][0], is_valid=False)
        files_src = opts.train['train']['src']
        files_tgt = opts.train['train']['tgt']
        self.data_train.read(files_src,files_tgt,token,self.vocab,max_length=opts.train['max_length'],example=opts.cfg['example_format'])

        logging.info('Read Valid data')
        self.data_valid = DataSet(opts.train['batch_size'][1], is_valid=True)
        files_src = opts.train['valid']['src']
        files_tgt = opts.train['valid']['tgt']
        self.data_valid.read(files_src,files_tgt,token,self.vocab,max_length=0,example=opts.cfg['example_format'])


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






