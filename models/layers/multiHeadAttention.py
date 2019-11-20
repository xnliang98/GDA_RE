import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

def clones(module, N):
    """ Produce N identical layers.
        Args:
            module (nn.Module): layers need to be copy.
            N (int): copy times.
        returns:
            nn.ModuleList of N identical module.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, mask=None, dropout=None):
    """ Compute 'Scaled Dot Product Attention'.
        Args:
            query (torch.tensor): Q
            key (torch.tensor): K
            mask (torch.tensor): mask of sequence
            dropout (float): dropout probability.
        returns:
            Attention(Q, K), Attention weights
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask==0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
   
    return p_attn

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
      
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.attn = None
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, mask=None):
        if mask is not None:
            # same mask applied to all h heads
            mask = mask.unsqueeze(1)
        nbatchs = query.size(0)
        
            # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key = [l(x).reshape(nbatchs, -1, self.h, self.d_k).transpose(1, 2)\
                for l, x in zip(self.linears, (query, key))]
            
            # 2) Apply attention on all the projected vectors in batch
        attn = attention(query, key, mask, self.dropout)

        return attn

if __name__ == "__main__":
    query = key = torch.rand(64, 100, 200)
    model = MultiHeadAttention(4, 200)
    attn = model(query, key)
    print(attn.shape)
    t = torch.split(attn, 1, dim=1)[0].squeeze(1)
    adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn, 1, dim=1)]
    print(t.shape)