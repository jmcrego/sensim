import io
import os
import sys
import gzip
import torch
import logging
import pyonmttok
import numpy as np
import json
import six
import random
#from torch.nn.utils.rnn import pad_sequence
from random import shuffle
from collections import defaultdict

idx_pad = 0
idx_unk = 1
idx_bos = 2
idx_eos = 3
idx_msk = 4
idx_sep = 5
idx_cls = 6
str_pad = '<pad>'
str_unk = '<unk>'
str_bos = '<bos>'
str_eos = '<eos>'
str_msk = '<msk>'
str_sep = '<sep>'
str_cls = '<cls>'


####################################################################
### OpenNMTTokenizer ###############################################
####################################################################
class OpenNMTTokenizer():

  def __init__(self, **kwargs):
    if 'mode' not in kwargs:
       logging.error('error: missing mode in tokenizer')
       sys.exit()
    mode = kwargs["mode"]
    del kwargs["mode"]
    self.tokenizer = pyonmttok.Tokenizer(mode, **kwargs)
    logging.info('built tokenizer mode={} {}'.format(mode,kwargs))

  def tokenize(self, text):
    tokens, _ = self.tokenizer.tokenize(text)
    return tokens

  def detokenize(self, tokens):
    return self.tokenizer.detokenize(tokens)


####################################################################
### Vocab ##########################################################
####################################################################
class Vocab():

    def __init__(self, file):
        self.tok_to_idx = {} # dict
        self.idx_to_tok = [] # vector
        self.idx_to_tok.append(str_pad)
        self.tok_to_idx[str_pad] = len(self.tok_to_idx) #0
        self.idx_to_tok.append(str_unk)
        self.tok_to_idx[str_unk] = len(self.tok_to_idx) #1
        self.idx_to_tok.append(str_bos)
        self.tok_to_idx[str_bos] = len(self.tok_to_idx) #2
        self.idx_to_tok.append(str_eos)
        self.tok_to_idx[str_eos] = len(self.tok_to_idx) #3
        self.idx_to_tok.append(str_msk)
        self.tok_to_idx[str_msk] = len(self.tok_to_idx) #4
        self.idx_to_tok.append(str_sep)
        self.tok_to_idx[str_sep] = len(self.tok_to_idx) #5
        self.idx_to_tok.append(str_cls)
        self.tok_to_idx[str_cls] = len(self.tok_to_idx) #6

        self.idx_pad = idx_pad
        self.idx_unk = idx_unk
        self.idx_bos = idx_bos
        self.idx_eos = idx_eos
        self.idx_msk = idx_msk
        self.idx_sep = idx_sep
        self.idx_cls = idx_cls

        with io.open(file, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
            for line in f:
                tok = line.strip()
                if tok not in self.tok_to_idx:
                    self.idx_to_tok.append(tok)
                    self.tok_to_idx[tok] = len(self.tok_to_idx)
        logging.info('read Vocab ({} entries) file={}'.format(len(self),file))

    def is_reserved(self, s):
        return s<7

    def __len__(self):
        return len(self.idx_to_tok)

    def __iter__(self):
        for tok in self.idx_to_tok:
            yield tok

    def __contains__(self, s): ### implementation of the method used when invoking : entry in vocab
        if type(s) == int: ### testing an index
            return s>=0 and s<len(self)
        ### testing a string
        return s in self.tok_to_idx

    def __getitem__(self, s): ### implementation of the method used when invoking : vocab[entry]
        if type(s) == int: ### input is an index, i want the string
            if s not in self:
                logging.error("key \'{}\' not found in vocab".format(s))
                sys.exit()
            return self.idx_to_tok[s]
            ### input is a string, i want the index
        if s not in self: 
            return idx_unk
        return self.tok_to_idx[s]

####################################################################
### batch ##########################################################
####################################################################

class batch():
    def __init__(self):
        self.idx_src = []
        self.idx_tgt = []
        self.lsrc = []
        self.ltgt = []
        self.src = []
        self.tgt = []
        self.sign = []
        self.maxlsrc = 0
        self.maxltgt = 0

    def __len__(self):
        return len(self.idx_src)

    def add_single(self, src, idx_src): ### used for pre-training (MLM): uses <cls> <bos> ... <eos>
        self.src.append(src)
        self.tgt.append([])
        self.idx_tgt.append([])
        self.ltgt.append(0)
        self.sign.append(0.0)

        idx_src.insert(0,idx_bos)
        idx_src.append(idx_eos)
        idx_src.insert(0,idx_cls)

        self.lsrc.append(len(idx_src))
        if len(idx_src) < self.maxlsrc:
            idx_src += [idx_pad]*(self.maxlsrc-len(idx_src)) 
        elif len(idx_src) > self.maxlsrc:
            self.maxlsrc = len(idx_src)
            for i in range(len(self.idx_src)):
                self.idx_src[i] += [idx_pad]*(self.maxlsrc-len(self.idx_src[i]))
        self.idx_src.append(idx_src) ### [<cls>, <bos>, <s1>, <s2>, ..., <sn>, <eos>, <pad>, ...]   (lsrc is the position of <eos> +1)

    def add_pair_join(self, src, idx_src, tgt, idx_tgt, train_swap): ### used for pre-training (MLM): uses <cls>, <bos> ... <eos> <sep> <bos> ... <eos>
        if train_swap and random.random() < 0.5:
            aux = list(tgt)
            tgt = list(src)
            src = list(aux)
            idx_aux = list(idx_tgt)
            idx_tgt = list(idx_src)
            idx_src = list(idx_aux)

        self.src.append(src)
        self.tgt.append(tgt)
        self.idx_tgt.append([])
        self.ltgt.append(0)
        self.sign.append(0.0)

        idx_src.insert(0,idx_bos)
        idx_src.append(idx_eos)
        idx_src.insert(0,idx_cls)

        idx_tgt.insert(0,idx_bos)
        idx_tgt.append(idx_eos)
        idx_tgt.insert(0,idx_sep)

        idx_src += idx_tgt

        self.lsrc.append(len(idx_src))
        if len(idx_src) < self.maxlsrc:
            idx_src += [idx_pad]*(self.maxlsrc-len(idx_src)) 
        elif len(idx_src) > self.maxlsrc:
            self.maxlsrc = len(idx_src)
            for i in range(len(self.idx_src)):
                self.idx_src[i] += [idx_pad]*(self.maxlsrc-len(self.idx_src[i]))
        self.idx_src.append(idx_src) ### [<cls>, <bos>, <s1>, <s2>, ..., <sn>, <eos>, <sep>, <bos>, <s1>, <s2>, ..., <sn>, <eos>, <pad>, ...]   (lsrc is the position of last <eos> +1)

    def add_pair(self, src, idx_src, tgt, idx_tgt, sign, train_swap): ### used for fine-tunning (SIM): uses <cls>, <bos> ... <eos> in both sides
        if train_swap and random.random() < 0.5:
            aux = list(tgt)
            tgt = list(src)
            src = list(aux)
            idx_aux = list(idx_tgt)
            idx_tgt = list(idx_src)
            idx_src = list(idx_aux)

        self.src.append(src)
        self.tgt.append(tgt)

        self.sign.append(sign)

        idx_src.insert(0,idx_bos)
        idx_src.append(idx_eos)
        idx_src.insert(0,idx_cls)

        idx_tgt.insert(0,idx_bos)
        idx_tgt.append(idx_eos)
        idx_tgt.insert(0,idx_cls)

        self.lsrc.append(len(idx_src))
        if len(idx_src) < self.maxlsrc:
            idx_src += [idx_pad]*(self.maxlsrc-len(idx_src)) 
        elif len(idx_src) > self.maxlsrc:
            self.maxlsrc = len(idx_src)
            for i in range(len(self.idx_src)):
                self.idx_src[i] += [idx_pad]*(self.maxlsrc-len(self.idx_src[i]))
        self.idx_src.append(idx_src) ### [<cls>, <bos>, <s1>, <s2>, ..., <sn>, <eos>, <pad>, ...]   (lsrc is the position of <eos> +1)

        self.ltgt.append(len(idx_tgt))
        if len(idx_tgt) < self.maxltgt:
            idx_tgt += [idx_pad]*(self.maxltgt-len(idx_tgt)) 
        elif len(idx_tgt) > self.maxltgt:
            self.maxltgt = len(idx_tgt)
            for i in range(len(self.idx_tgt)):
                self.idx_tgt[i] += [idx_pad]*(self.maxltgt-len(self.idx_tgt[i]))
        self.idx_tgt.append(idx_tgt) ### [<cls>, <bos>, <t1>, <t2>, ..., <tn>, <eos>, <pad>, ...]   (ltgt is the position of <eos> +1)

####################################################################
### Dataset ########################################################
####################################################################

class Dataset():

    def __init__(self, token, vocab, max_length=0):
        self.token = token
        self.vocab = vocab
        self.max_length = max_length
        self.idx = []
        self.len = []

    def addfiles(self, fsrc, ftgt, src_bos=True, src_eos=True, src_cls=True, src_sep=False, tgt_bos=True, tgt_eos=True, tgt_cls=False, tgt_sep=True):
        if fsrc.endswith('.gz'): 
            fs = gzip.open(fsrc, 'rb')
        else: 
            fs = io.open(fsrc, 'r', encoding='utf-8', newline='\n', errors='ignore')
        if ftgt.endswith('.gz'): 
            ft = gzip.open(ftgt, 'rb')
        else: 
            ft = io.open(ftgt, 'r', encoding='utf-8', newline='\n', errors='ignore')

        n_filt = 0
        for ls, lt in zip(fs,ft):
            sidx = [self.vocab[s] for s in self.token.tokenize(ls)]
            tidx = [self.vocab[t] for t in self.token.tokenize(lt)]
            if len(sidx)+len(tidx) > self.max_length:
                n_filt += 1
                continue

            if src_bos:
                sidx.insert(0,idx_bos)
            if src_eos:
                sidx.append(idx_eos)
            if src_cls:
                sidx.insert(0,idx_cls)
            if src_sep:
                sidx.insert(0,idx_sep)

            if tgt_bos:
                tidx.insert(0,idx_bos)
            if tgt_eos:
                tidx.append(idx_eos)
            if tgt_cls:
                tidx.insert(0,idx_cls)
            if tgt_sep:
                tidx.insert(0,idx_sep)

            self.len.append(len(sidx)+len(tidx))
            self.idx.append([sidx,tidx])

        logging.info('found {} sentences in files: [{},{}] #filtered: {}'.format(len(self.idx),fsrc,ftgt,n_filt))


    def addfile(self, file, add_bos=True, add_eos=True, add_cls=True, add_sep=False):
        if file.endswith('.gz'): 
            f = gzip.open(file, 'rb')

        n_filt = 0
        for l in f:
            idx = [self.vocab[t] for t in self.token.tokenize(l)]
            if len(idx) > self.max_length:
                n_filt += 1
                continue

            if add_bos:
                idx.insert(0,idx_bos)
            if add_eos:
                idx.append(idx_eos)
            if add_cls:
                idx.insert(0,idx_cls)
            if add_sep:
                idx.insert(0,idx_sep)

            self.idx.append([idx])
            self.len.append(len(idx))
        logging.info('found {} sentences in file:{} #filtered: {}'.format(len(self.idx),file,n_filt))


    def buildbatches(self, batch_size, p_swap=0.0, p_uneven=0.0, allow_shuffle=False):
        self.batches = []
        indexs = [i for i in range(len(self.idx))] #indexs in original order
        if allow_shuffle:
            logging.debug('sorting data to minimize padding')
            indexs = np.argsort(self.len)




    def __len__(self):
        return len(self.idx)


####################################################################
### DataSet ########################################################
####################################################################

class DataSet():

    def __init__(self, steps, files, token, vocab, sim_run=False, batch_size=32, max_length=0, p_uneven=0.0, swap_bitext=False, allow_shuffle=False, is_infinite=False):
        self.allow_shuffle = allow_shuffle
        self.is_infinite = is_infinite
        self.max_length = max_length
        self.batch_size = batch_size
        self.steps = steps
        self.sim_run = sim_run
        self.p_uneven = p_uneven
        self.swap_bitext = swap_bitext
        logging.info('reading dataset [swap:{},batch_size:{},max_length:{},sim_run:{},allow_shuffle:{},is_infinite:{}]'.format(swap_bitext,batch_size,max_length,sim_run,allow_shuffle,is_infinite))
        ##################
        ### read files ###
        ##################
        max_num_sents = 0 ### jmcc
        self.data = []
        for i in range(len(files)):
            if len(files[i])==1: ############# single file ##########################################
                fsrc = files[i][0]
                if self.sim_run: ### skip when fine-tuning on similarity
                    logging.info('skip single file: {}'.format(fsrc))
                    continue
                if fsrc.endswith('.gz'): fs = gzip.open(fsrc, 'rb')
                else: fs = io.open(fsrc, 'r', encoding='utf-8', newline='\n', errors='ignore')
                n = 0
                m = 0
                for ls in fs:
                    n += 1
                    src = [s for s in token.tokenize(ls)]
                    if self.max_length > 0 and len(src) > self.max_length: 
                        continue
                    m += 1
                    self.data.append([src,[]]) ### [s1, s2, ..., sn], [t1, t2, ..., tn]
                    if max_num_sents > 0 and m >= max_num_sents: break
                logging.info('read {} out of {} sentences from file [{}]'.format(m,n,fsrc))
            else: ############################ two files ############################################
                fsrc = files[i][0]
                ftgt = files[i][1]
                if fsrc.endswith('.gz'): fs = gzip.open(fsrc, 'rb')
                else: fs = io.open(fsrc, 'r', encoding='utf-8', newline='\n', errors='ignore')
                if ftgt.endswith('.gz'): ft = gzip.open(ftgt, 'rb')
                else: ft = io.open(ftgt, 'r', encoding='utf-8', newline='\n', errors='ignore')
                n = 0
                m = 0
                for ls, lt in zip(fs,ft):
                    n += 1
                    src = [s for s in token.tokenize(ls)]
                    tgt = [t for t in token.tokenize(lt)]
                    if self.max_length > 0 and len(src)+len(tgt) > self.max_length: 
                        continue
                    m += 1
                    self.data.append([src,tgt]) ### [s1, s2, ..., sn], [t1, t2, ..., tn]
                    if max_num_sents > 0 and m >= max_num_sents: break
                logging.info('read {} out of {} sentences from files [{},{}]'.format(m,n,fsrc,ftgt))
        logging.info('read {} examples'.format(len(self.data)))
        #####################
        ### build batches ###
        #####################
        self.batches = []
        indexs = [i for i in range(len(self.data))] #indexs in original order
        if self.allow_shuffle:
            logging.debug('sorting data to minimize padding')
            data_len = [len(x[0])+len(x[1]) for x in self.data]
            indexs = np.argsort(data_len) #indexs sorted by length of data

        currbatch = batch()
        for i in range(len(indexs)):
            index = indexs[i]
            #print('\nindex',index)
            src = self.data[index][0]
            #print('src',src)
            idx_src = [vocab[s] for s in src]
            #print('idx_src',idx_src)
            if self.sim_run: ### fine tunning (SIM) 
                if random.random() < self.p_uneven and i > 0:
                    sign = 1.0 ### NOT parallel (divergent)
                    index = indexs[i-1]
                else:
                    sign = -1.0 ### parallel (not divergent)
                    index = indexs[i]
                #print('index',index)
                #print('sign',sign)
                tgt = self.data[index][1]
                #print('tgt',tgt)
                idx_tgt = [vocab[t] for t in tgt]
                #print('idx_tgt',idx_tgt)
                currbatch.add_pair(src,idx_src,tgt,idx_tgt,sign,swap_bitext)
            else: ### pre-training (MLM)
                if len(self.data[index]) > 1:
                    tgt = self.data[index][1]
                    idx_tgt = [vocab[t] for t in tgt]
                    currbatch.add_pair_join(src,idx_src,tgt,idx_tgt,swap_bitext)
                else:
                    currbatch.add_single(src,idx_src)
            if len(currbatch) == self.batch_size or i == len(indexs)-1: ### record new batch
                self.batches.append(currbatch)
                currbatch = batch()
        logging.info('built {} batches'.format(len(self.batches)))


    def __iter__(self):

        indexs = [i for i in range(len(self.batches))]
        while True: 

            if self.allow_shuffle: 
                logging.debug('shuffling batches')
                shuffle(indexs)

            for index in indexs:
                yield self.batches[index]

            if not self.is_infinite:
                break









