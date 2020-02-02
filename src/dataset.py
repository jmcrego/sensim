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
from torch.nn.utils.rnn import pad_sequence
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
        self.isParallel = []
        self.maxlsrc = 0
        self.maxltgt = 0

    def __len__(self):
        return len(self.idx_src)

    def add_single(self, src, idx_src): ### used for pre-training (MLM): uses <cls> <bos> ... <eos>
        self.src.append(src)
        self.tgt.append([])
        self.idx_tgt.append([])
        self.ltgt.append(0)
        self.isParallel.append(0.0)

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
        self.isParallel.append(0.0)

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


    def add_pair(self, src, idx_src, tgt, idx_tgt, isParallel): ### used for fine-tunning (SIM): uses <cls>, <bos> ... <eos> in both sides
        if train_swap and random.random() < 0.5:
            aux = list(tgt)
            tgt = list(src)
            src = list(aux)
            idx_aux = list(idx_tgt)
            idx_tgt = list(idx_src)
            idx_src = list(idx_aux)

        self.src.append(src)
        self.tgt.append(tgt)

        self.isParallel.append(isParallel)

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
### DataSet ########################################################
####################################################################

class DataSet():

    def __init__(self, steps, files, token, vocab, batch_size=32, max_length=0,swap_bitext=False, allow_shuffle=False, valid_test=False):
        self.allow_shuffle = allow_shuffle
        self.valid_test = valid_test

        self.max_length = max_length
        self.batch_size = batch_size
        self.steps = steps
        self.do_sim = self.steps['sim']['run']
        self.p_uneven = self.steps['sim']['p_uneven']
        self.swap_bitext = swap_bitext
        logging.info('reading dataset [swap:{},batch_size:{},max_length:{},do_sim:{},allow_shuffle:{},valid_test:{}]'.format(swap_bitext,batch_size,max_length,self.do_sim,allow_shuffle,valid_test))
        ##################
        ### read files ###
        ##################
        max_num_sents = 0 ### jmcrego
        self.data = []
        for i in range(len(files)):
            if len(files[i])==1: ############# single file ##########################################
                fsrc = files[i][0]
                if self.do_sim: ### skip when fine-tuning on similarity
                    logging.info('skip single file: {}'.format(fsrc))
                    continue
                if fsrc.endswith('.gz'): fs = gzip.open(fsrc, 'rb')
                else: fs = io.open(fsrc, 'r', encoding='utf-8', newline='\n', errors='ignore')
                n = 0
                m = 0
                for ls in fs:
                    n += 1
                    src = [s for s in token.tokenize(ls)]
                    if not self.valid_test and self.max_length > 0 and len(src) > self.max_length: 
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
                    if not self.valid_test and self.max_length > 0 and len(src)+len(tgt) > self.max_length: 
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
            src = self.data[index][0]
            idx_src = [vocab[s] for s in src]
            if self.do_sim: ### fine tunning (SIM) 
                if random.random() < self.p_uneven and i > 0:
                    isParallel = -1.0 ### NOT parallel
                    index = indexs[i-1]
                else:
                    isParallel = 1.0 ### parallel
                    index = indexs[i]
                tgt = self.data[index][1]
                idx_tgt = [vocab[t] for t in tgt]
                currbatch.add_pair(src,idx_src,tgt,idx_tgt,isParallel,swap_bitext)
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

            if not self.valid_test:
                break









