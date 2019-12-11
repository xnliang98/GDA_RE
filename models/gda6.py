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
from models.layers import MultiHeadAttention, PositionAwareAttention, PAAttention
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
        self.pe_emb = nn.Embedding(constant.MAX_LEN * 2 + 1, opt['pe_emb'])
        self.rnn = MyRNN(input_size, opt['hidden_dim'] // 2, opt['rnn_layer'],
            bidirectional=True, dropout=opt['rnn_dropout'], use_cuda=opt['cuda'])
        self.input_W_G = nn.Linear(input_size, self.mem_dim)
        self.num_blocks = 3
        self.blks = nn.ModuleList()
        for i in range(self.num_blocks):
            self.blks.append(MyBlock(opt, self.mem_dim, dropout=0.2))
        self.drop = nn.Dropout(opt['rnn_dropout'])
        # self.pos_attn = PositionAwareAttention(self.mem_dim, self.mem_dim, opt['pe_emb'] * 2, self.mem_dim)
        self.linear = nn.Linear(self.mem_dim * self.num_blocks, self.mem_dim)

        self.out = nn.Linear(self.mem_dim, self.mem_dim)
        self.classifier = nn.Linear(opt['hidden_dim'], opt['num_class'])

        self.init_embeddings()

    def init_embeddings(self):
        if self.opt['pe_emb'] > 0:
            self.pe_emb.weight.data.uniform_(-1.0, 1.0)
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
        src_mask = (words != constant.PAD_ID).unsqueeze(-2)
        word_embs = self.emb(words)
        embs = [word_embs]

        if self.opt['pos_dim'] > 0:
            embs += [self.pos_emb(pos)]
        if self.opt['ner_dim'] > 0:
            embs += [self.ner_emb(ner)]
        
        embs = torch.cat(embs, dim=2)
        embs = self.drop(embs)
     
        # inputs, hidden = self.rnn(embs, masks)
        inputs = self.input_W_G(embs)

        # def inputs_to_tree_reps(head, l):
        #     trees = [head_to_tree(head[i], l[i]) for i in range(len(l))]
        #     adj = [tree_to_adj(maxlen, tree, directed=False).reshape(1, maxlen, maxlen) for tree in trees]
        #     adj = np.concatenate(adj, axis=0)
        #     adj = torch.from_numpy(adj)
        #     return adj.cuda() if self.opt['cuda'] else adj

        # adj = inputs_to_tree_reps(head.data, l)

        def inputs_to_tree_reps(head, words, l, prune, subj_pos, obj_pos):
            head, words, subj_pos, obj_pos = head.cpu().numpy(), words.cpu().numpy(), subj_pos.cpu().numpy(), obj_pos.cpu().numpy()
            trees = [head_to_tree(head[i], words[i], l[i], prune, subj_pos[i], obj_pos[i]) for i in range(len(l))]
            adj = [tree_to_adj(maxlen, tree, directed=False, self_loop=False).reshape(1, maxlen, maxlen) for tree in trees]
            adj = np.concatenate(adj, axis=0)
            adj = torch.from_numpy(adj)
            return adj.cuda() if self.opt['cuda'] else adj
        adj = inputs_to_tree_reps(head.data, words.data, l, 1, subj_pos.data, obj_pos.data)

        subj_pe_inputs = self.pe_emb(subj_pos + constant.MAX_LEN)
        obj_pe_inputs = self.pe_emb(obj_pos + constant.MAX_LEN)
        pe_features = torch.cat((subj_pe_inputs, obj_pe_inputs), dim=2)
        
        # gcn_outputs, _ = self.gcn(adj, inputs)
        # gcn_outputs = self.drop(gcn_outputs)
        outputs_list = []
        outputs = inputs
        for i in range(self.num_blocks):
            outputs, _, _ = self.blks[i](outputs, adj, pe_features, masks)
            outputs_list.append(outputs)
        # outputs = self.linear(torch.cat(outputs_list, -1))
        outputs = F.relu(self.out(torch.sum(outputs, dim=1).squeeze()))
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

class MyBlock(nn.Module):
    def __init__(self, opt, mem_dim, dropout):
        super(MyBlock, self).__init__()
        self.mem_dim = mem_dim
        self.gcn = GCNLayer(self.mem_dim, self.mem_dim, opt['second_layer'], opt['gcn_dropout'])
        self.lstm = MyRNN(self.mem_dim, self.mem_dim // 2, 1, bidirectional=True, use_cuda=opt['cuda'])
        self.linear = nn.Linear(mem_dim, mem_dim)
        self.pa_attn = PAAttention(self.mem_dim, self.mem_dim, opt['pe_emb'] * 2, self.mem_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, inputs, adj, pe_features, masks):
        lstm_outputs, hidden = self.lstm(inputs, masks)
        gcn_outputs, _ = self.gcn(adj, lstm_outputs)
        gcn_outputs = self.dropout(gcn_outputs)
        hidden = self.dropout(torch.cat([hidden[-1, :, :], hidden[-2, :, :]], dim=-1))
        out1 = attention(gcn_outputs, lstm_outputs, lstm_outputs, masks)
        outputs = self.pa_attn(lstm_outputs, masks, hidden, pe_features, gcn_outputs)
        return outputs, gcn_outputs, hidden
    

        