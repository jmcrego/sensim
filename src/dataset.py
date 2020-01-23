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

    def __init__(self, files, token, vocab, batch_size, max_length, sim_uneven=0.0, ali_uneven=0.0, allow_shuffle=False, single_epoch=False):
        self.allow_shuffle = allow_shuffle
        self.single_epoch = single_epoch
        self.batches = []
        data_msk = [] ### msk examples from bitexts
        data_sim = [] ### sim examples from bitexts
        data_ali = [] ### ali examples from bitexts
        data_mon = [] ### msk examples from monolingual data
        max_num_examples = 1000 #this is only used for debugging (set to 0 to avoid)
        for l in range(len(files)):
            if len(files[l])==2:
                fsrc = files[l][0]
                ftgt = files[l][1]
                ### src
                if fsrc.endswith('.gz'): 
                    fs = gzip.open(fsrc, 'rb')
                else: 
                    fs = io.open(fsrc, 'r', encoding='utf-8', newline='\n', errors='ignore')
                ### tgt
                if ftgt.endswith('.gz'): 
                    ft = gzip.open(ftgt, 'rb')
                else: 
                    ft = io.open(ftgt, 'r', encoding='utf-8', newline='\n', errors='ignore')

                n = 0
                m = 0
                tgt2_idx = []
                for ls, lt in zip(fs,ft):
                    n += 1
                    src_idx = [vocab[s] for s in token.tokenize(ls)]
                    tgt_idx = [vocab[t] for t in token.tokenize(lt)]
                    if max_length > 0 and (len(src_idx) + len(tgt_idx)) > max_length: 
                        continue
                    m += 1
                    #src sentence
                    snt_src_idx = []
                    snt_src_idx.append(idx_bos)
                    snt_src_idx.extend(src_idx)
                    snt_src_idx.append(idx_eos)
                    #tgt sentence
                    snt_tgt_idx = []
                    snt_tgt_idx.append(idx_bos)
                    snt_tgt_idx.extend(tgt_idx)
                    snt_tgt_idx.append(idx_eos)
                    #tgt2 sentence (not parallel)
                    snt_tgt2_idx = []
                    snt_tgt2_idx.append(idx_bos)
                    snt_tgt2_idx.extend(tgt2_idx)
                    snt_tgt2_idx.append(idx_eos)
                    #src_tgt sentence
                    snt_src_tgt_idx = []
                    snt_src_tgt_idx.append(idx_cls)
                    snt_src_tgt_idx.extend(snt_src_idx)
                    snt_src_tgt_idx.append(idx_sep)
                    snt_src_tgt_idx.extend(snt_tgt_idx)
                    #src_tgt2 sentence
                    snt_src_tgt2_idx = []
                    snt_src_tgt2_idx.append(idx_cls)
                    snt_src_tgt2_idx.extend(snt_src_idx)
                    snt_src_tgt2_idx.append(idx_sep)
                    snt_src_tgt2_idx.extend(snt_tgt2_idx)

                    ### add to data lists
                    data_msk.append(snt_src_tgt_idx) ### [<cls>, <bos>, <s1>, <s2>, ..., <sn>, <eos>, <sep>, <bos>, <t1>, <t2>, ..., <tn>, <eos>]

                    if random.random() < sim_uneven and len(tgt2_idx): 
                        data_sim.append([snt_src_tgt2_idx, -1.0]) ### [<cls>, <bos>, <s1>, <s2>, ..., <sn>, <eos>, <sep>, <bos>, <t1>, <t2>, ..., <tn>, <eos>], -1.0
                    else:
                        data_sim.append([snt_src_tgt_idx, 1.0]) ### [<cls>, <bos>, <s1>, <s2>, ..., <sn>, <eos>, <sep>, <bos>, <t1>, <t2>, ..., <tn>, <eos>], 1.0

                    if random.random() < ali_uneven and len(tgt2_idx):
                        data_ali.append([snt_src_idx, snt_tgt2_idx, -1.0]) ### [<bos>, <s1>, <s2>, ..., <sn>, <eos>], [<bos>, <t1>, <t2>, ..., <tn>, <eos>], -1.0
                    else:
                        data_ali.append([snt_src_idx, snt_tgt_idx, 1.0]) ### [<bos>, <s1>, <s2>, ..., <sn>, <eos>], [<bos>, <t1>, <t2>, ..., <tn>, <eos>], 1.0

                    tgt2_idx = tgt_idx
                    if max_num_examples > 0 and m >= max_num_examples: break
                logging.info('read {} out of {} sentence pairs from files [{}, {}]'.format(m,n,fsrc,ftgt))
            else:
                fsrc = files[l][0]
                ### src
                if fsrc.endswith('.gz'): 
                    fs = gzip.open(fsrc, 'rb')
                else: 
                    fs = io.open(fsrc, 'r', encoding='utf-8', newline='\n', errors='ignore')

                n = 0
                m = 0
                for ls in fs:
                    n += 1
                    src_idx = [vocab[s] for s in token.tokenize(ls)]
                    if max_length > 0 and len(src_idx) > max_length: 
                        continue
                    m += 1
                    snt_idx = []
                    snt_idx.append(idx_cls)
                    snt_idx.append(idx_bos)
                    snt_idx.extend(src_idx)
                    snt_idx.append(idx_eos)
                    data_mon.append(snt_idx) ### [<cls>, <bos>, <w1>, <w2>, ..., <wn>, <eos>]
                    if max_num_examples > 0 and m >= max_num_examples: break
                logging.info('read {} out of {} sentences from file [{}]'.format(m,n,fsrc))
        logging.info('read {} single sentences, {} sentence pairs'.format(len(data_mon), len(data_msk)+len(data_sim)+len(data_ali)))
        ###
        ### building batches with all data read
        ###
        ### mon 
        ###
        indexs = [i for i in range(len(data_mon))] #indexs in original order
        if self.allow_shuffle:
            logging.debug('sorting data according to sentence size to minimize padding')
            data_len = [len(x) for x in data_mon]
            indexs = np.argsort(data_len) #indexs sorted by length of data

        curr_batch = []
        for i in range(len(indexs)):
            curr_batch.append(data_mon[indexs[i]])
            if len(curr_batch) == batch_size or i == len(indexs)-1: #full batch or last example
                #add padding
                max_len = max([len(s) for s in curr_batch])
                for k in range(len(curr_batch)):
                    curr_batch[k] += [idx_pad]*(max_len-len(curr_batch[k]))
                self.batches.append(['mon', curr_batch, [], []]) #step, batch, batch_tgt, batch_isparallel
                curr_batch = []
        ###
        ### msk 
        ###
        indexs = [i for i in range(len(data_msk))] #indexs in original order
        if self.allow_shuffle:
            logging.debug('sorting data according to sentence size to minimize padding')
            data_len = [len(x) for x in data_msk]
            indexs = np.argsort(data_len) #indexs sorted by length of data

        curr_batch = []
        for i in range(len(indexs)):
            curr_batch.append(data_msk[indexs[i]])
            if len(curr_batch) == batch_size or i == len(indexs)-1: #full batch or last example
                #add padding
                max_len = max([len(s) for s in curr_batch])
                for k in range(len(curr_batch)):
                    curr_batch[k] += [idx_pad]*(max_len-len(curr_batch[k]))
                self.batches.append(['msk', curr_batch, [], []]) #step, batch, batch_tgt, batch_isparallel
                curr_batch = []
        ###
        ### sim
        ###
        indexs = [i for i in range(len(data_sim))] #indexs in original order
        if self.allow_shuffle:
            logging.debug('sorting data according to sentence size to minimize padding')
            data_len = [len(x[0]) for x in data_sim]
            indexs = np.argsort(data_len) #indexs sorted by length of sentences

        curr_batch = []
        curr_batch_isparallel = []
        for i in range(len(indexs)):
            curr_batch.append(data_sim[indexs[i]][0])
            curr_batch_isparallel.append(data_sim[indexs[i]][1])
            if len(curr_batch) == batch_size or i == len(indexs)-1: #full batch or last example
                #add padding
                max_len = max([len(s) for s in curr_batch])
                for k in range(len(curr_batch)):
                    curr_batch[k] += [idx_pad]*(max_len-len(curr_batch[k]))
                self.batches.append(['sim', curr_batch, [], curr_batch_isparallel]) #step, batch, batch_tgt, batch_isparallel
                curr_batch = []
                curr_batch_isparallel = []
        ###
        ### ali
        ###
        indexs = [i for i in range(len(data_ali))] #indexs in original order
        if self.allow_shuffle:
            logging.debug('sorting data according to sentence size to minimize padding')
            data_len = [len(x[0]) for x in data_ali]
            indexs = np.argsort(data_len) #indexs sorted by length of source sentence

        curr_batch_src = []
        curr_batch_tgt = []
        curr_batch_isparallel = []
        for i in range(len(indexs)):
            curr_batch_src.append(data_ali[indexs[i]][0])
            curr_batch_tgt.append(data_ali[indexs[i]][1])
            curr_batch_isparallel.append(data_ali[indexs[i]][2])
            if len(curr_batch_src) == batch_size or i == len(indexs)-1: #full batch or last example
                #add padding
                max_len_src = max([len(s) for s in curr_batch_src])
                max_len_tgt = max([len(t) for t in curr_batch_tgt])
                for k in range(len(curr_batch_src)):
                    curr_batch_src[k] += [idx_pad]*(max_len_src-len(curr_batch_src[k]))
                    curr_batch_tgt[k] += [idx_pad]*(max_len_tgt-len(curr_batch_tgt[k]))
                self.batches.append(['ali', curr_batch_src, curr_batch_tgt, curr_batch_isparallel]) #step, batch, batch_tgt, batch_isparallel
                curr_batch_src = []
                curr_batch_tgt = []
                curr_batch_isparallel = []


    def __len__(self):
        return len(self.batches)


    def __iter__(self):
        while True: #### infinite loop
            indexs = [i for i in range(len(self.batches))] #indexs in original order
            if self.allow_shuffle: 
                logging.debug('shuffling batches in dataset')
                shuffle(indexs) #indexs randomized
            ### iterate 
            for i in indexs:
                yield self.batches[i]
            ### end of epoch
            logging.info('end of dataset epoch')
            if self.single_epoch:
                return



