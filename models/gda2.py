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
        self.pe_emb = nn.Embedding(constant.MAX_LEN * 2 + 1, opt['pe_emb'])
        self.rnn = MyRNN(input_size, opt['hidden_dim'] // 2, opt['rnn_layer'],
            bidirectional=True, dropout=opt['rnn_dropout'], use_cuda=opt['cuda'])
 
        self.blk1 = firstBlock(opt, self.mem_dim, self.pe_emb, 0.5)
        self.blk2 = secondBlock(opt, self.mem_dim, self.pe_emb, 0.5)
        self.blk3 = secondBlock(opt, self.mem_dim, self.pe_emb, 0.5)
        self.blk4 = secondBlock(opt, self.mem_dim, self.pe_emb, 0.5)
        
        self.in_drop = nn.Dropout(opt['in_dropout'])
        self.rnn_drop = nn.Dropout(opt['rnn_dropout'])
        self.drop = nn.Dropout(opt['rnn_dropout'])

        self.layer0 = nn.Linear(self.mem_dim * 4, self.mem_dim * 2)
        self.layer1 = nn.Linear(self.mem_dim * 2, self.mem_dim)
        self.layer2 = nn.Linear(self.mem_dim * 3, self.mem_dim)
        self.classifier = nn.Linear(opt['hidden_dim'], opt['num_class'])
        self.attn = MultiHeadAttention(opt['heads'], self.mem_dim)
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
        embs = self.in_drop(embs)
     
        inputs = self.rnn_drop(self.rnn(embs, masks)[0])
        # inputs = self.input_W_G(inputs)

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

        gcn_masks = (adj.sum(1) + adj.sum(2)).eq(0).unsqueeze(2)
        out1 = self.blk1(inputs, adj, subj_pos, obj_pos, masks)

        attn_tensor = self.attn(out1, out1, src_mask)
        attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]
        out2 = self.blk2(out1, attn_adj_list, subj_pos, obj_pos, masks)

        attn_tensor = self.attn(out2, out2, src_mask)
        attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]
        out3 = self.blk3(out2, attn_adj_list, subj_pos, obj_pos, masks)
        
        attn_tensor = self.attn(out3, out3, src_mask)
        attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]
        out4 = self.blk3(out3, attn_adj_list, subj_pos, obj_pos, masks)

        outputs = torch.cat([out1, out2, out3, out4], dim=-1)
        # h = self.drop(outputs)
        h = F.relu(self.layer0(outputs))
        # h = self.drop(h)
        h = F.relu(self.layer1(h))

        h_out = pool(h, gcn_masks, "max")
        subj_mask, obj_mask = subj_pos.eq(0).eq(0).unsqueeze(2), obj_pos.eq(0).eq(0).unsqueeze(2)
        subj_out = pool(h, subj_mask, "max")
        obj_out = pool(h, obj_mask, "max")
        outputs = self.in_drop(torch.cat([h_out, subj_out, obj_out], dim=1))

        outputs = F.relu(self.drop(self.layer2(outputs)))
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

class firstBlock(nn.Module):
    def __init__(self, opt, mem_dim, pe_emb, dropout):
        super(firstBlock, self).__init__()
        self.mem_dim = mem_dim
        self.gcn1 = DenseGCN(self.mem_dim, opt['first_layer'], opt['gcn_dropout'])
        # self.gcn2 = DenseGCN(self.mem_dim // 2, opt['second_layer'], opt['gcn_dropout'])
        self.pe_emb = pe_emb
        self.in_lstm = nn.Linear(self.mem_dim + opt['pe_emb'] * 2, self.mem_dim)
        self.lstm = MyRNN(self.mem_dim, self.mem_dim // 2, 2, bidirectional=True, use_cuda=opt['cuda'])
        self.linear = nn.Linear(mem_dim, mem_dim)
        # self.out = nn.Linear(mem_dim, mem_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, inputs, adj, subj_pos, obj_pos, masks):
        gcn_outputs1 = self.gcn1(adj, inputs)
        # gcn_outputs2 = self.gcn2(adj, inputs[:, :, self.mem_dim // 2:])
        # gcn_outputs = torch.cat([gcn_outputs1, gcn_outputs2], dim=-1)
        gcn_outputs = gcn_outputs1
        rnn_inputs = inputs
        
        subj_pe_inputs = self.pe_emb(subj_pos + constant.MAX_LEN)
        obj_pe_inputs = self.pe_emb(obj_pos + constant.MAX_LEN)
        pe_features = torch.cat((subj_pe_inputs, obj_pe_inputs), dim=2)
        rnn_inputs = self.in_lstm(torch.cat([pe_features, inputs], dim=-1))
        lstm_outputs, _ = self.lstm(rnn_inputs, masks)
        # lstm_outputs = self.layer1(lstm_outputs)
        h1, h2 = gcn_outputs, lstm_outputs
    
        # h = torch.cat([h1, h2], dim=-1)
        h = self.dropout(attention(h1, h2, h2, masks))
        h = self.linear(h)

        return h + inputs

class secondBlock(nn.Module):
    def __init__(self, opt, mem_dim, pe_emb, dropout):
        super(secondBlock, self).__init__()
        self.mem_dim = mem_dim
        self.gcn = MultiDenseGCN(opt['heads'], self.mem_dim, opt['first_layer'], opt['gcn_dropout'])
        self.pe_emb = pe_emb
        self.in_lstm = nn.Linear(self.mem_dim + opt['pe_emb'] * 2, self.mem_dim)
        self.lstm = MyRNN(self.mem_dim, self.mem_dim // 2, 2, bidirectional=True, use_cuda=opt['cuda'])
        self.linear = nn.Linear(mem_dim, mem_dim)
        # self.out = nn.Linear(mem_dim, mem_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, adj_list, subj_pos, obj_pos, masks):
        gcn_outputs = self.gcn(adj_list, inputs)
        rnn_inputs = inputs
        
        subj_pe_inputs = self.pe_emb(subj_pos + constant.MAX_LEN)
        obj_pe_inputs = self.pe_emb(obj_pos + constant.MAX_LEN)
        pe_features = torch.cat((subj_pe_inputs, obj_pe_inputs), dim=2)
        rnn_inputs = self.in_lstm(torch.cat([pe_features, inputs], dim=-1))
        lstm_outputs, _ = self.lstm(rnn_inputs, masks)
        h1, h2 = gcn_outputs, lstm_outputs

        # h = torch.cat([h1, h2], dim=-1)
        h = self.dropout(attention(h1, h2, h2, masks))
        h = self.linear(h)
        return h + inputs


    

        