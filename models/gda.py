"""
AGGCNs model for relation extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
# from torch.nn.modules import MultiheadAttention

from models.layers import pool, MyRNN, DenseGCN, MultiDenseGCN, GCNLayer
from models.layers import MultiHeadAttention
from models.layers import Tree, head_to_tree, tree_to_adj
from utils import constant, torch_utils

class GDAClassifier(nn.Module):
    def __init__(self, opt, emb_matrix=None):
        super(GDAClassifier, self).__init__()
        self.opt = opt
        self.emb_matrix = emb_matrix
        
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt['pos_dim']) if opt['pos_dim'] > 0 else None
        self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), opt['ner_dim']) if opt['ner_dim'] > 0 else None
        
        self.in_dim = opt['emb_dim']
        if self.pos_emb:
            self.in_dim += opt['pos_dim']
        if self.ner_emb:
            self.in_dim += opt['ner_dim']
        
        self.mem_dim = opt['hidden_dim']
        input_size = self.in_dim
        self.rnn = MyRNN(input_size, opt['hidden_dim'], opt['rnn_layer'],
            bidirectional=True, dropout=opt['rnn_dropout'], use_cuda=opt['cuda'])
        self.in_dim = opt['hidden_dim'] * 2
        
        
        self.gcn1 = GCNLayer(self.mem_dim, self.mem_dim, opt['first_layer'], opt['gcn_dropout'])
        self.gcn2 = GCNLayer(self.mem_dim, self.mem_dim, opt['second_layer'], opt['gcn_dropout'])
        if opt['pe_emb'] > 0:
            self.pe_emb = nn.Embedding(constant.MAX_LEN * 2 + 1, opt['pe_emb'])
            self.in_lstm = nn.Linear(self.mem_dim * 2 + opt['pe_emb'] * 2, self.mem_dim)
        self.lstm = MyRNN(self.mem_dim, self.mem_dim, 1, 
            bidirectional=True, dropout=opt['rnn_dropout'], use_cuda=opt['cuda'])
        self.in_drop = nn.Dropout(opt['in_dropout'])
        self.rnn_drop = nn.Dropout(opt['rnn_dropout'])

        self.layer0 = nn.Linear(self.mem_dim * 2, self.mem_dim)
        self.layer1 = nn.Linear(self.mem_dim * 2, self.mem_dim)
        self.layer2 = nn.Linear(self.mem_dim * 2, self.mem_dim * 2)
        self.layer3 = nn.Linear(self.mem_dim * 2, self.mem_dim * 2)
        self.layer4 = nn.Linear(self.mem_dim * 4, self.mem_dim)
        self.classifier = nn.Linear(opt['hidden_dim'], opt['num_class'])

        self.init_embeddings()

    def init_embeddings(self):

        if self.emb_matrix is None:
            self.emb.weight.data[1:, :].uniform_(-1.0, 1.0)
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)
        
        if self.opt['pe_emb'] > 0:
            self.pe_emb.weight.data.uniform_(-1.0, 1.0)
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

     
        inputs = self.rnn_drop(self.rnn(embs, masks)[0])
        # inputs = self.input_W_G(inputs)

        def inputs_to_tree_reps(head, l):
            trees = [head_to_tree(head[i], l[i]) for i in range(len(l))]
            adj = [tree_to_adj(maxlen, tree, directed=False).reshape(1, maxlen, maxlen) for tree in trees]
            adj = np.concatenate(adj, axis=0)
            adj = torch.from_numpy(adj)
            return adj.cuda() if self.opt['cuda'] else adj

        adj = inputs_to_tree_reps(head.data, l)
        gcn_masks = (adj.sum(1) + adj.sum(2)).eq(0).unsqueeze(2)
        gcn_outputs1, _ = self.gcn1(adj, inputs[:, :, :self.mem_dim])
        gcn_outputs2, _ = self.gcn2(adj, inputs[:, :, self.mem_dim:])
        gcn_outputs = torch.cat([gcn_outputs1, gcn_outputs2], dim=-1)
        gcn_outputs = self.layer0(gcn_outputs)

        rnn_inputs = inputs
        if self.opt['pe_emb'] > 0:
            subj_pe_inputs = self.pe_emb(subj_pos + constant.MAX_LEN)
            obj_pe_inputs = self.pe_emb(obj_pos + constant.MAX_LEN)
            pe_features = torch.cat((subj_pe_inputs, obj_pe_inputs), dim=2)
            rnn_inputs = self.in_lstm(torch.cat([pe_features, inputs], dim=-1))
        lstm_outputs, _ = self.lstm(rnn_inputs, masks)
        lstm_outputs = self.layer1(lstm_outputs)
        h1, h2 = gcn_outputs, lstm_outputs
        # h = attention(gcn_outputs, lstm_outputs, gcn_outputs)
        h = torch.cat([h1, h2], dim=-1)
        h_head = self.layer2(h)
        h_tail = self.layer3(h)
        # h = attention(h, h, h, masks)
        
        # h_out = pool(h, gcn_masks, "max")
        subj_mask, obj_mask = subj_pos.eq(0).eq(0).unsqueeze(2), obj_pos.eq(0).eq(0).unsqueeze(2)
        subj_out = pool(h_head, subj_mask, "max")
        obj_out = pool(h_tail, obj_mask, "max")
        outputs = self.in_drop(torch.cat([subj_out, obj_out], dim=1))
        outputs = F.relu(self.layer4(outputs))
        outputs = self.classifier(outputs)
        return outputs



def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    mask = mask.unsqueeze(1)
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
        # print(scores)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    return torch.matmul(p_attn, value)