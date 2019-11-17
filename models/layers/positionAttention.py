'''
Position Aware Attention Layer.
'''
import math
import copy

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.nn import init
import numpy as np
from utils import constant, torch_utils


class PositionAwareAttention(nn.Module):
    """
    A position-augmented attention layer where the attention weight is
    a = T' . tanh(Ux + Vq + Wf)
    where x is the input, q is the query, and f is additional position features.
    """
    def __init__(self, input_size, query_size, feature_size, attn_size):
        super(PositionAwareAttention, self).__init__()

        self.input_size = input_size
        self.query_size = query_size
        self.feature_size = feature_size
        self.attn_size = attn_size
        self.ulinear = nn.Linear(input_size, attn_size)
        self.vlinear = nn.Linear(query_size, attn_size, bias=False)
        if feature_size > 0:
            self.wlinear = nn.Linear(feature_size, attn_size, bias=False)
        else:
            self.wlinear = None
        self.tlinear = nn.Linear(attn_size, 1)
        self.init_weight()
    
    def init_weight(self):
        init.normal_(self.ulinear.weight.data, std=0.001)
        init.normal_(self.vlinear.weight.data, std=0.001)
        if self.wlinear is not None:
            init.normal_(self.wlinear.weight.data, std=0.001)
        init.zeros_(self.tlinear.weight)
    
    def forward(self, x, x_mask, f):
        batch_size, seq_len, _ = x.size()

        x_proj = self.ulinear(x) # ==> B, T, A
        # q_proj = self.vlinear(q) # ==> B, T, A
    

        f_proj = self.wlinear(f) # ==> B, T, A
        projs = [x_proj, f_proj] 
     

        scores = self.tlinear(torch.tanh(sum(projs))).reshape(batch_size, seq_len) # B, T

        # mask padding
        scores.data.masked_fill_(x_mask.data, -1e9)
        weights = F.softmax(scores, dim=1) # ==> B, T
        # weights = weights.ge(0.5)
        # weighted average input vectors
        out = weights.unsqueeze(1).bmm(x).squeeze(1) # B, 1, T x B, T, I ==> B, 1, I ==> B, I
        return out