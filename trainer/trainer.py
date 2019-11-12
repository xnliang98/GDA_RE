"""
Base Trainer.
"""

import torch
from torch.nn import functional as F
from torch import nn

import numpy as np
from utils import torch_utils


def unpack_batch(batch, cuda):
    ''' Unpack batch which is load from data/loader.py
        if cuda, move variables from batch to cuda.
    '''
    if cuda:
        inputs = [item.cuda() for item in batch[:10]]
        labels = batch[10].cuda()
    else:
        inputs = [item for item in batch[:10]]
        labels = batch[10]
    tokens = batch[0]
    head = batch[5]
    subj_pos = batch[6]
    obj_pos = batch[7]
    lens = batch[1].eq(0).long().sum(1).squeeze()
    return inputs, labels, tokens, head, subj_pos, obj_pos, lens

class Trainer(object):
    """ Interface of Trainer of each model training. """
    def __init__(self, opt, emb_matrix=None):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)
    
    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Can not load model from {}.".format(filename))
            exit(1)
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']
    
    def save(self, filename, epoch):
        params = {
            'model': self.model.state_dict(),
            'config': self.opt,
        }
        try:
            torch.save(params, filename)
            print("Model saved to {}.".format(filename))
        except BaseException:
            print("[ Warning: Saving failed... continuing anyway. ]")

        
