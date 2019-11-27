import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from models.layers import MyRNN
from utils import constant, torch_utils

class GDAClassifier(nn.Module):
    def __init__(self, opt, emb_matrix=None):
        super(GDAClassifier, self).__init__()
        self.opt = opt
        self.emb_matrix = emb_matrix
        self.layers = opt['layers']
        # create embedding layers
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt['pos_dim']) if opt['pos_dim'] > 0 else None
        self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), opt['ner_dim']) if opt['ner_dim'] > 0 else None
        embeddings = (self.emb, self.pos_emb, self.ner_emb)
        self.in_dim = opt['emb_dim'] + opt['pos_dim'] + opt['ner_dim']
        self.in_drop = nn.Dropout(opt['input_dropout'])
        self.mem_dim = opt['hidden_dim']
     
        input_size = self.in_dim
        self.rnn = MyRNN(input_size, opt['hidden_dim'], 2,
            bidirectional=True, dropout=opt['rnn_dropout'], use_cuda=opt['cuda'])

        self.dropout = nn.Dropout(0.1)  # use on last layer output

        self.W_O = nn.Linear(self.mem_dim * 2, self.mem_dim)
        self.W_S = nn.Linear(self.mem_dim * 2, self.mem_dim)
        self.W_G = nn.Linear(self.mem_dim * 2, self.mem_dim)
        self.classifier = nn.Linear(opt['hidden_dim'], opt['num_class'])

        self.init_embeddings()

    def init_embeddings(self):

        if self.emb_matrix is None:
            self.emb.weight.data[1:, :].uniform_(-1.0, 1.0)
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)
        
        # decide finetuning
        if self.opt['topn'] <= 0:
            print("Do not finetune word embedding layer.")
            self.emb.weight.requires_grad = False
        elif self.opt['topn'] < self.opt['vocab_size']:
            print("Finetune top {} word embeddings.".format(self.opt['topn']))
            self.emb.weight.register_hook(lambda x: torch_utils.keep_partial_grad(x, self.opt['topn']))
        else:
            print("Finetune all embeddings.")

    def forward(self, inputs):
        words, masks, pos, ner, deprel, head, subj_pos, obj_pos, subj_type, obj_type = inputs # unpack
        l = (masks.data.cpu().numpy() == 0).astype(np.int64).sum(1)
        maxlen = max(l)
        
        word_embs = self.emb(words)
        embs = [word_embs]

        if self.opt['pos_dim'] > 0:
            embs += [self.pos_emb(pos)]
        if self.opt['ner_dim'] > 0:
            embs += [self.ner_emb(ner)]
        embs = torch.cat(embs, dim=2)
        embs = self.in_drop(embs)

        rnn_outputs, hidden = self.rnn(embs, masks)
        init_hidden1 = torch.zeros(rnn_outputs.size(0), rnn_outputs.size(-1)).cuda()
        init_hidden2 = torch.zeros(rnn_outputs.size(0), rnn_outputs.size(-1)).cuda()
        obj_hidden, subj_hidden = init_hidden1, init_hidden2
        for b in range(embs.size(0)):
            subj_index = (subj_pos[b] + masks[b].type(torch.long)).eq(0).unsqueeze(-1)
            obj_index = (masks[b].type(torch.long) + obj_pos[b]).eq(0).unsqueeze(-1)
            subjs = torch.masked_select(rnn_outputs[b], subj_index)
            objs = torch.masked_select(rnn_outputs[b], obj_index)
            if subjs.shape[0] > rnn_outputs.size(-1):
                subjs = subjs.reshape(-1, rnn_outputs.size(-1))
                subjs = torch.sum(subjs, dim=0).squeeze(0)
            if objs.shape[0] > rnn_outputs.size(-1):
                objs = objs.reshape(-1, rnn_outputs.size(-1))
                objs = torch.sum(objs, dim=0).squeeze(0)
            obj_hidden[b] = objs
            subj_hidden[b] = subjs
        
        global_hidden = torch.cat([hidden[-1], hidden[-2]], dim=-1)

        obj, _ = attention(obj_hidden, rnn_outputs, rnn_outputs, mask=masks, dropout=self.dropout)
        subj, _ = attention(subj_hidden, rnn_outputs, rnn_outputs, mask=masks, dropout=self.dropout)
        glob, _ = attention(global_hidden, rnn_outputs, rnn_outputs, mask=masks, dropout=self.dropout)
        obj = self.W_O(obj)
        subj = self.W_S(subj)
        glob = self.W_G(glob)
        outputs = F.relu(obj + subj + glob).squeeze()
        outputs = self.classifier(outputs)
        return outputs

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    query = query.unsqueeze(1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    mask = mask.unsqueeze(1)
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    return torch.matmul(p_attn, value), p_attn
