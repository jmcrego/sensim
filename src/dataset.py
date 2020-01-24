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
        dist = 0.0
        if 'msk_mon' in self.steps and 'dist' in self.steps['msk_mon']: dist = self.steps['msk_mon']['dist']
        if dist == 0.0: return

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
            curr_batch.append(snt_idx)
            if len(curr_batch) == self.batch_size or i == len(indexs)-1: #full batch or last example                
                self.batches_mon.append(self.add_padding(curr_batch)) 
                curr_batch = []


    def build_msk_batches(self):
        self.batches_msk = []
        dist = 0.0
        if 'msk_par' in self.steps and 'dist' in self.steps['msk_par']: dist = self.steps['msk_par']['dist']
        if dist == 0.0: return

        indexs = [i for i in range(len(self.data_btxt))] #indexs in original order
        if self.allow_shuffle:
            logging.debug('sorting data_msk to minimize padding')
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
            curr_batch.append(snt_idx)
            if len(curr_batch) == self.batch_size or i == len(indexs)-1: #full batch or last example                
                self.batches_msk.append(self.add_padding(curr_batch)) 
                curr_batch = []


    def build_sim_batches(self):
        self.batches_sim = []
        dist = 0.0
        if 'sim' in self.steps and 'dist' in self.steps['sim']: dist = self.steps['sim']['dist']
        if dist == 0.0: return

        p_uneven = self.steps['sim']['p_uneven']
        indexs = [i for i in range(len(self.data_btxt))] #indexs in original order
        if self.allow_shuffle:
            logging.debug('sorting data_sim to minimize padding')
            data_len = [len(x[0])+len(x[1]) for x in self.data_btxt]
            indexs = np.argsort(data_len) #indexs sorted by length of data
        curr_batch = []
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
            snt_idx = []
            snt_idx.append(idx_cls)
            snt_idx.extend(src_idx)
            snt_idx.append(idx_sep)
            snt_idx.extend(tgt_idx)
            curr_batch.append(snt_idx)
            curr_batch_isparallel.append(isparallel)
            if len(curr_batch) == self.batch_size or i == len(indexs)-1: #full batch or last example                
                self.batches_sim.append([self.add_padding(curr_batch),curr_batch_isparallel]) 
                curr_batch = []
                curr_batch_isparallel = []


    def build_ali_batches(self):
        self.batches_ali = []
        dist = 0.0
        if 'ali' in self.steps and 'dist' in self.steps['ali']: dist = self.steps['ali']['dist']
        if dist == 0.0: return


        p_uneven = self.steps['ali']['p_uneven']
        indexs = [i for i in range(len(self.data_btxt))] #indexs in original order
        if self.allow_shuffle:
            logging.debug('sorting data_ali to minimize padding')
            data_len = [len(x[0]) for x in self.data_btxt]
            indexs = np.argsort(data_len) #indexs sorted by length of data
        curr_batch_src = []
        curr_batch_tgt = []
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
            curr_batch_isparallel.append(isparallel)
            if len(curr_batch_src) == self.batch_size or i == len(indexs)-1: #full batch or last example                
                self.batches_ali.append([self.add_padding(curr_batch_src), self.add_padding(curr_batch_tgt), curr_batch_isparallel]) 
                curr_batch_src = []
                curr_batch_tgt = []
                curr_batch_isparallel = []


    def __init__(self, steps, files, token, vocab, batch_size=32, max_length=0, allow_shuffle=False, infinite=False):
        self.allow_shuffle = allow_shuffle
        self.infinite = infinite
        self.max_length = max_length
        self.batch_size = batch_size
        self.steps = steps
        self.steps_run = defaultdict(int)
        self.read_data(files,token,vocab,0) #1000 is only used for debugging (delete to avoid filtering)
        self.build_mon_batches()
        self.build_msk_batches()
        self.build_sim_batches()
        self.build_ali_batches()


    def __iter__(self):
        ### if is validation/test set then i loop once over all the examples
        ### no need to shuffle
        if not self.infinite: 
            for i in range(len(self.batches_mon)):
                yield 'mon', self.batches_mon[i]
                self.steps_run['mon'] += 1
            for i in range(len(self.batches_msk)):
                yield 'msk', self.batches_msk[i]
                self.steps_run['msk'] += 1
            for i in range(len(self.batches_sim)):
                yield 'sim', self.batches_sim[i]
                self.steps_run['ali'] += 1
            for i in range(len(self.batches_ali)):
                yield 'ali', self.batches_ali[i]
                self.steps_run['ali'] += 1
            return

        ### if training i loop forever follwing the distributions indicated by self.steps['dist']
        dist_mon = self.steps['msk_mon']['dist']
        if dist_mon > 0.0 and len(self.batches_mon) == 0:
            logging.error('no batches_mon entries and dist_mon={:.2f}'.format(dist_mon))
            sys.exit()

        dist_msk = self.steps['msk_par']['dist']
        if dist_msk > 0.0 and len(self.batches_msk) == 0:
            logging.error('no batches_msk entries and dist_msk={:.2f}'.format(dist_msk))
            sys.exit()

        dist_sim = self.steps['sim']['dist']
        if dist_sim > 0.0 and len(self.batches_sim) == 0:
            logging.error('no batches_sim entries and dist_sim={:.2f}'.format(dist_sim))
            sys.exit()

        dist_ali = self.steps['ali']['dist']
        if dist_ali > 0.0 and len(self.batches_ali) == 0:
            logging.error('no batches_ali entries and dist_ali={:.2f}'.format(dist_ali))
            sys.exit()

        indexs_mon = [i for i in range(len(self.batches_mon))]
        indexs_msk = [i for i in range(len(self.batches_msk))]
        indexs_sim = [i for i in range(len(self.batches_sim))]
        indexs_ali = [i for i in range(len(self.batches_ali))]
        ### do shuffle if required
        if self.allow_shuffle: 
            shuffle(indexs_mon)
            shuffle(indexs_msk)
            shuffle(indexs_sim)
            shuffle(indexs_ali)
        i_mon = 0
        i_msk = 0
        i_sim = 0
        i_ali = 0
        while True: #### infinite loop
            p = random.random() #[0.0, 1.0)
            if p < dist_mon:
                if i_mon >= len(indexs_mon):
                    i_mon = 0
                yield 'mon', self.batches_mon[indexs_mon[i_mon]]
                self.steps_run['mon'] += 1
                i_mon += 1

            elif p < dist_mon+dist_msk:
                if i_msk >= len(indexs_msk):
                    i_msk = 0
                yield 'msk', self.batches_msk[indexs_msk[i_msk]]
                self.steps_run['msk'] += 1
                i_msk += 1

            elif p < dist_mon+dist_msk+dist_sim:
                if i_sim >= len(indexs_sim):
                    i_sim = 0
                yield 'sim', self.batches_sim[indexs_sim[i_sim]]
                self.steps_run['sim'] += 1
                i_sim += 1

            elif p < dist_mon+dist_msk+dist_sim+dist_ali:
                if i_ali >= len(indexs_ali):
                    i_ali = 0
                yield 'ali', self.batches_ali[indexs_ali[i_ali]]
                self.steps_run['ali'] += 1
                i_ali += 1

    def info(self):
        total = 0
        for step in self.steps_run:
            total += self.steps_run[step]
        return '[mon:{:.2f} par:{:.2f} sim:{:.2f} ali:{:.2f}]'.format(self.steps_run['mon']/total, self.steps_run['msk']/total, self.steps_run['sim']/total, self.steps_run['ali']/total)


