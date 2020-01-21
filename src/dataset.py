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

    def __init__(self, batch_size, is_valid=False):
        self.data = []
        self.data_len = []
        self.batches = []
        self.batch_size = batch_size
        self.is_valid = is_valid

    def read(self, lsrc, ltgt, token_src, token_tgt, vocab, max_length=0, example=None):
        for l in range(len(lsrc)):
            ### src
            if lsrc[l].endswith('.gz'): 
                fs = gzip.open(lsrc[l], 'rb')
            else: 
                fs = io.open(lsrc[l], 'r', encoding='utf-8', newline='\n', errors='ignore')
            ### tgt
            if ltgt[l].endswith('.gz'): 
                ft = gzip.open(ltgt[l], 'rb')
            else: 
                ft = io.open(ltgt[l], 'r', encoding='utf-8', newline='\n', errors='ignore')

            n = 0
            m = 0
            for ls, lt in zip(fs,ft):
                n += 1
                #print(ls)
                #print(lt)
                src_idx = [vocab[s] for s in token_src.tokenize(ls)]
                tgt_idx = [vocab[t] for t in token_tgt.tokenize(lt)]
                if max_length > 0 and (len(src_idx) + len(tgt_idx)) > max_length: 
                    continue
                m += 1
                ### src-side
                if 'src' in example and 'cls' in example['src'] and example['src']['cls']:
                    src_idx.insert(0,idx_cls)
                if 'src' in example and 'bos' in example['src'] and example['src']['bos']:
                    src_idx.insert(0,idx_bos)
                if 'src' in example and 'eos' in example['src'] and example['src']['eos']:
                    src_idx.append(idx_eos)
                ### separator
                if 'sep' in example and example['sep']:
                    src_idx.append(idx_sep)
                ### tgt-side
                if 'tgt' in example and 'cls' in example['tgt'] and example['tgt']['cls']:
                    tgt_idx.insert(0,idx_cls)
                if 'tgt' in example and 'bos' in example['tgt'] and example['tgt']['bos']:
                    tgt_idx.insert(0,idx_bos)
                if 'tgt' in example and 'eos' in example['tgt'] and example['tgt']['eos']:
                    tgt_idx.append(idx_eos)

                snt_idx = src_idx + tgt_idx
                #print(snt_idx)
                self.data.append(snt_idx)
                self.data_len.append(len(snt_idx))
            logging.info('read {} out of {} sentences in src/tgt files [{}, {}]'.format(m,n,lsrc[l],ltgt[l]))
        logging.info('read {} sentences'.format(len(self.data)))
        return


    def build_batches(self): 
        self.batches = []
        indexs = [i for i in range(len(self.data_len))]
        if not self.is_valid:
            logging.debug('sorting data according to sentence size to minimize padding')
            indexs = np.argsort(self.data_len)

        logging.debug('building batches')
        batch, batch_len = [], []
        for i in range(len(indexs)):
            ### add to batch
            batch.append(self.data[indexs[i]])
            batch_len.append(len(batch[-1]))
            ### batch filled or last example
            if len(batch) == self.batch_size or i == len(indexs)-1: 
                ### add padding
                for k in range(len(batch)):
                    batch[k] += [idx_pad]*(max(batch_len)-len(batch[k]))
                ### add batch
                self.batches.append([batch, batch_len])
                batch, batch_len = [], []

        logging.info('found {} batches in {} sentences of dataset'.format(len(self.batches),len(indexs)))


    def __len__(self):
        return len(self.batches)


    def __iter__(self):
        while True: #### infinite loop
            indexs = [i for i in range(len(self.batches))]
            ### randomize batches if not is_valid
            if not self.is_valid: 
                logging.debug('shuffling batches in dataset')
                shuffle(indexs)
            ### iterate 
            for i in indexs:
                yield self.batches[i]
            if self.is_valid:
                return




