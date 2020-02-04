import io
import gzip
import numpy as np
import torch
import logging
import time
import random
import sys
import glob
import os
import pyonmttok
from collections import defaultdict
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from src.dataset import Vocab, DataSet, OpenNMTTokenizer, batch
from src.trainer import sequence_mask
from src.model import make_model
from src.optim import NoamOpt, LabelSmoothing, CosineSIM, AlignSIM, ComputeLossMLM, ComputeLossSIM


class Infer():

    def __init__(self, opts):
        self.dir = opts.dir
        self.vocab = Vocab(opts.cfg['vocab'])
        self.cuda = opts.cfg['cuda']
        V = len(self.vocab)
        N = opts.cfg['num_layers']
        d_model = opts.cfg['hidden_size']
        d_ff = opts.cfg['feedforward_size']
        h = opts.cfg['num_heads']
        dropout = opts.cfg['dropout']
        self.token = OpenNMTTokenizer(**opts.cfg['token'])

        self.model = make_model(V, N=N, d_model=d_model, d_ff=d_ff, h=h, dropout=dropout)
        if self.cuda:
            self.model.cuda()

        ### load checkpoint
        files = sorted(glob.glob(self.dir + '/checkpoint.???????.pth')) 
        if len(files):
            file = files[-1] ### last is the newest
            checkpoint = torch.load(file)
            self.model.load_state_dict(checkpoint['model'])
            logging.info('loaded checkpoint {}'.format(file))
#        else:
#            logging.error('no checkpoint available')
#            sys.exit()


    def __call__(self, fsrc, pooling='cls'):
        logging.info('Start testing')
        if fsrc.endswith('.gz'): fs = gzip.open(fsrc, 'rb')
        else: fs = io.open(fsrc, 'r', encoding='utf-8', newline='\n', errors='ignore')

        self.data = []
        self.model.eval()
        with torch.no_grad():
            for ls in fs:
                src = [s for s in token.tokenize(ls)]
                idx_src = [self.vocab[s] for s in src]
                idx_src.insert(0,idx_bos)
                idx_src.append(idx_eos)
                idx_src.insert(0,idx_cls)
                x = torch.from_numpy([idx_src]) #[batch_size, max_len] the original words with padding
                x_mask = torch.as_tensor((x != self.vocab.idx_pad)).unsqueeze(-2) #[batch_size, 1, max_len]
                mask_s = torch.from_numpy(sequence_mask([len(idx_src)],mask_n_initials=2))

                if self.cuda:
                    x.cuda()
                    x_mask.cuda()
                    mask_s.cuda()

                h = self.model.forward(x,x_mask)
                if pooling == 'max':
                    s, _ = torch.max(h*mask_s + (1.0-mask_s)*-999.9, dim=1) #-999.9 should be -Inf but it produces an nan when multiplied by 0.0
                elif pooling == 'mean':
                    s = torch.sum(h * mask_s, dim=1) / torch.sum(mask_s, dim=1)
                elif pooling == 'cls':
                    s = h[:, 0, :] # take embedding of first token <cls>
                print(s)

        logging.info('End validation')


