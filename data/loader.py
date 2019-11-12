"""
Data loader for TACRED json files.
"""

import json
import random
import os
import numpy as np
import torch
from utils import constant, helper, vocab


class DataLoader(object):
    """
    load data from json file, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, vocab, evaluation=False):
        self.batch_size = batch_size
        self.opt = opt
        self.vocab = vocab 
        self.eval = evaluation
        self.label2id = constant.LABEL_TO_ID
        # load data from json file
        with open(filename) as fin:
            data = json.load(fin)
        self.raw_data = data
        data = self.preprocess(data, vocab, opt)

        # shuffle for trianing 
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.labels = [self.id2label[d[-1]] for d in data]
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i: i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        print("{} batches created for {}".format(len(data), filename))
    
    def preprocess(self, data, vocab, opt):
        """preprocess data and convert to word ids.
            return :
            [(
                tokens # tokens ids with padding,
                pos # pos ids
                ner # ner ids
                deprel # deprel ids
                head # head
                subj_positions # relative positions of subj
                obj_positions # relative positions of obj
                subj_type # subj ner type
                obj_type # obj ner type
                relation # relation id
            ), ...]
        """
        processed = []
        for d in data:
            tokens = list(d['token'])
            if opt['lower']:
                tokens = [t.lower() for t in tokens]
            
            # get obj/subj position
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']
            # replace obj/subj with obj/subj+type
            tokens[ss: se+1] = ['SUBJ-' + d['subj_type']] * (se - ss + 1)
            tokens[os: oe+1] = ['OBJ-' + d['obj_type']] * (oe - os + 1)
            # get list of ids of tokens
            tokens = map_to_ids(tokens, vocab.word2id)
            # pos and ner and deprel to id
            pos = map_to_ids(d['stanford_pos'], constant.POS_TO_ID)
            ner = map_to_ids(d['stanford_ner'], constant.NER_TO_ID)
            deprel = map_to_ids(d['stanford_deprel'], constant.DEPREL_TO_ID)
            head = [int(i) for i in d['stanford_head']]
            assert any([x == 0 for x in head])
            l = len(tokens)
            # compute positions
            subj_positions = get_positions(ss, se, l)
            obj_positions = get_positions(os, oe, l)
            subj_type = [constant.SUBJ_NER_TO_ID[d['subj_type']]]
            obj_type = [constant.OBJ_NER_TO_ID[d['obj_type']]]
            relation = self.label2id[d['relation']]
            processed += [(tokens, pos, ner, deprel, head, subj_positions, obj_positions, subj_type, obj_type, relation)]
        return processed
    
    def gold(self):
        return self.labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, key):
        """
        Get a batch of tensor with index.
            (words, 
            masks, # mask of padding
            pos, 
            ner, 
            deprel, 
            head, 
            subj_positions, 
            obj_positions, 
            subj_type, 
            obj_type, # obj ner type
            rels, # relation id 
            orig_idx # unsorted idx
            )
        """
        if not isinstance(key, int):
            raise TypeError
        if key < 0  or key >= len(self.data):
            raise TypeError

        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        assert len(batch) == 10

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)

        # word dropout
        if not self.eval:
            words = [word_dropout(sent, self.opt['word_dropout']) for sent in batch[0]]
        else:
            words = batch[0]
        
        # convert to tensors
        words = get_long_tensor(words, batch_size)
        masks = torch.eq(words, 0) # mask of padding
        pos = get_long_tensor(batch[1], batch_size)
        ner = get_long_tensor(batch[2], batch_size)
        deprel = get_long_tensor(batch[3], batch_size)
        head = get_long_tensor(batch[4], batch_size)
        subj_positions = get_long_tensor(batch[5], batch_size)
        obj_positions = get_long_tensor(batch[6], batch_size)
        subj_type = get_long_tensor(batch[7], batch_size)
        obj_type = get_long_tensor(batch[8], batch_size)

        rels = torch.LongTensor(batch[9])

        return (words, masks, pos, ner, deprel, head, subj_positions, obj_positions, subj_type, obj_type, rels, orig_idx)
    
    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
    
def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids

def get_positions(start_idx, end_idx, length):
    """Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [0] * (end_idx - start_idx + 1) \
        + list(range(1, length - end_idx))

def get_long_tensor(tokens_list, batch_size):
    """ Convert batch of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(constant.PAD_ID)
    for i, s in enumerate(tokens_list):
        tokens[i, : len(s)] = torch.LongTensor(s)
    return tokens

def sort_all(batch, lens):
    """ Sort all fields bt descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2: ], sorted_all[1]

def word_dropout(tokens, dropout):
    """ Randomly dropout token from tokens with UNK"""
    return [constant.UNK_ID if x != constant.UNK_ID and np.random.random() < dropout else x for x in tokens]