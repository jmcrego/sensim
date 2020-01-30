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
    #print(tokens)
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
### DataSet ########################################################
####################################################################
class DataSet():

    def read_data(self, files, token, vocab, max_num_examples=0):
        self.data_mono = []
        self.data_btxt = []
        for i in range(len(files)):
            if len(files[i])==1:
                file = files[i][0]
                if file.endswith('.gz'): f = gzip.open(file, 'rb')
                else: f = io.open(file, 'r', encoding='utf-8', newline='\n', errors='ignore')
                n = 0
                m = 0
                for ls in f:
                    n += 1
                    src_idx = [vocab[s] for s in token.tokenize(ls)]
                    if self.max_length > 0 and len(src_idx) > self.max_length: 
                        continue
                    m += 1
                    src_idx.insert(0,idx_bos)
                    src_idx.append(idx_eos)
                    self.data_mono.append(src_idx) ### [<bos>, <w1>, <w2>, ..., <wn>, <eos>]
                    if max_num_examples > 0 and m >= max_num_examples: break
                logging.info('read {} out of {} sentences from file [{}]'.format(m,n,file))
            else:
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
                    src_idx = [vocab[s] for s in token.tokenize(ls)]
                    tgt_idx = [vocab[t] for t in token.tokenize(lt)]
                    if self.max_length > 0 and len(src_idx)+len(tgt_idx) > self.max_length: 
                        continue
                    m += 1
                    src_idx.insert(0,idx_bos)
                    src_idx.append(idx_eos)
                    tgt_idx.insert(0,idx_bos)
                    tgt_idx.append(idx_eos)
                    self.data_btxt.append([src_idx,tgt_idx]) ### [<bos>, <s1>, <s2>, ..., <sn>, <eos>], [<bos>, <t1>, <t2>, ..., <tn>, <eos>]
                    if max_num_examples > 0 and m >= max_num_examples: break
                logging.info('read {} out of {} sentences from files [{},{}]'.format(m,n,fsrc,ftgt))
        logging.info('read {} mono sentences'.format(len(self.data_mono)))
        logging.info('read {} btxt sentences'.format(len(self.data_btxt)))


    def add_padding(self, batch):
        #batch is a list of lists that may all be equally sized (using <pad>)
        max_len = max([len(l) for l in batch])
        for i in range(len(batch)):
            batch[i] += [idx_pad]*(max_len-len(batch[i]))
        return batch


    def build_mon_batches(self):
        self.batches_mon = []
        indexs = [i for i in range(len(self.data_mono))] #indexs in original order
        if self.allow_shuffle:
            logging.debug('sorting data_mon to minimize padding')
            data_len = [len(x) for x in self.data_mono]
            indexs = np.argsort(data_len) #indexs sorted by length of data
        curr_batch = []
        for i in range(len(indexs)):
            index = indexs[i]
            snt_idx = self.data_mono[index]
            snt_idx.insert(0,idx_cls)
            curr_batch.append(snt_idx) #<cls>, <bos>, <s1>, <s2>, ..., <sn>, <eos>
            if len(curr_batch) == self.batch_size or i == len(indexs)-1: #full batch or last example                
                self.batches_mon.append(self.add_padding(curr_batch)) 
                curr_batch = []
        logging.info('built {} mon batches'.format(len(self.batches_mon)))


    def build_par_batches(self):
        self.batches_par = []
        indexs = [i for i in range(len(self.data_btxt))] #indexs in original order
        if self.allow_shuffle:
            logging.debug('sorting data_par to minimize padding')
            data_len = [len(x[0])+len(x[1]) for x in self.data_btxt]
            indexs = np.argsort(data_len) #indexs sorted by length of data
        curr_batch = []
        for i in range(len(indexs)):
            index = indexs[i]
            src_idx = self.data_btxt[index][0]
            tgt_idx = self.data_btxt[index][1]
            snt_idx = []
            snt_idx.append(idx_cls)
            snt_idx.extend(src_idx)
            snt_idx.append(idx_sep)
            snt_idx.extend(tgt_idx)
            curr_batch.append(snt_idx) #<cls> <bos>, <s1>, <s2>, ..., <sn>, <eos>, <sep>, <bos>, <t1>, <t2>, ..., <tn>, <eos>
            if len(curr_batch) == self.batch_size or i == len(indexs)-1: #full batch or last example                
                self.batches_par.append(self.add_padding(curr_batch)) 
                curr_batch = []
        logging.info('built {} par batches'.format(len(self.batches_par)))


    def build_sim_batches(self):
        self.batches_sim = []
        p_uneven = self.steps['sim']['p_uneven']
        indexs = [i for i in range(len(self.data_btxt))] #indexs in original order
        if self.allow_shuffle:
            logging.debug('sorting data_btxt to minimize padding')
            data_len = [len(x[0]) for x in self.data_btxt]
            indexs = np.argsort(data_len) #indexs sorted by length of data
        curr_batch_src = []
        curr_batch_tgt = []
        curr_batch_src_len = []
        curr_batch_tgt_len = []
        curr_batch_isparallel = []
        for i in range(len(indexs)):
            index = indexs[i]
            src_idx = self.data_btxt[index][0]
            tgt_idx = self.data_btxt[index][1]
            isparallel = 1.0 ### parallel
            if random.random() < p_uneven and i>0:
                index_prev = indexs[i-1]
                tgt_idx = self.data_btxt[index_prev][1]
                isparallel = -1.0 ### NOT parallel
            curr_batch_src.append(src_idx)
            curr_batch_tgt.append(tgt_idx)
            curr_batch_src_len.append(len(src_idx))
            curr_batch_tgt_len.append(len(tgt_idx))
            curr_batch_isparallel.append(isparallel)
            if len(curr_batch_src) == self.batch_size or i == len(indexs)-1: #full batch or last example
                self.batches_sim.append([self.add_padding(curr_batch_src), self.add_padding(curr_batch_tgt), curr_batch_src_len, curr_batch_tgt_len, curr_batch_isparallel]) 
                curr_batch_src = []
                curr_batch_tgt = []
                curr_batch_src_len = []
                curr_batch_tgt_len = []
                curr_batch_isparallel = []
        logging.info('built {} sim batches'.format(len(self.batches_sim)))


    def __init__(self, steps, files, token, vocab, batch_size=32, max_length=0, allow_shuffle=False, infinite=False):
        self.allow_shuffle = allow_shuffle
        self.infinite = infinite
        self.max_length = max_length
        self.batch_size = batch_size
        self.steps = steps
        self.read_data(files,token,vocab,10000) #1000 is only used for debugging (delete to avoid filtering)

        self.dist_mon = self.steps['mon']['dist']
        self.dist_par = self.steps['par']['dist']
        self.run_sim = self.steps['sim']['run']
        if self.run_sim:
            self.build_sim_batches()
        else:
            if self.dist_mon > 0.0:
                self.build_mon_batches()
            if self.dist_par > 0.0:
                self.build_par_batches()


    def __iter__(self):
        ### if is validation/test set then i loop once over all the examples
        ### no need to shuffle
        if not self.infinite: 
            for i in range(len(self.batches_mon)):
                yield 'mon', self.batches_mon[i]
            for i in range(len(self.batches_par)):
                yield 'par', self.batches_par[i]
            for i in range(len(self.batches_sim)):
                yield 'sim', self.batches_sim[i]
            return

        ### if training i loop forever following the distributions indicated by self.steps['mon']['dist'] and self.steps['par']['dist']
        if self.run_sim:
            if len(self.batches_sim) == 0:
                logging.error('no batches_sim entries and run_sim={}'.format(run_sim))
                sys.exit()
            indexs_sim = [i for i in range(len(self.batches_sim))]
            if self.allow_shuffle: 
                shuffle(indexs_sim)

        else:
            if self.dist_mon > 0.0 and len(self.batches_mon) == 0:
                logging.error('no batches_mon entries and dist_mon={:.2f}'.format(self.dist_mon))
                sys.exit()

            if self.dist_par > 0.0 and len(self.batches_par) == 0:
                logging.error('no batches_par entries and dist_par={:.2f}'.format(self.dist_par))
                sys.exit()
            indexs_mon = [i for i in range(len(self.batches_mon))]
            indexs_par = [i for i in range(len(self.batches_par))]
            if self.allow_shuffle: 
                shuffle(indexs_mon)
                shuffle(indexs_par)

        i_mon = 0
        i_par = 0
        i_sim = 0
        while True: #### infinite loop
            if not self.run_sim:
                p = random.random() #[0.0, 1.0)
                if p < self.dist_mon:
                    if i_mon >= len(indexs_mon):
                        i_mon = 0
                    yield 'mon', self.batches_mon[indexs_mon[i_mon]]
                    i_mon += 1

                elif p < self.dist_mon+self.dist_par:
                    if i_par >= len(indexs_par):
                        i_par = 0
                    yield 'par', self.batches_par[indexs_par[i_par]]
                    i_par += 1
            else:
                if i_sim >= len(indexs_sim):
                    i_sim = 0
                yield 'sim', self.batches_sim[indexs_sim[i_sim]]
                i_sim += 1










