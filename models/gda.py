"""
AGGCNs model for relation extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.layers import pool, MyRNN, DenseGCN, MultiDenseGCN
from models.layers import MultiHeadAttention, attention, PositionAwareAttention
from models.layers import Tree, head_to_tree, tree_to_adj
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
        
        self.mem_dim = opt['hidden_dim']
        if opt.get('rnn', False):
            input_size = self.in_dim
            self.rnn = MyRNN(input_size, opt['hidden_dim'], opt['rnn_layers'],
                bidirectional=True, dropout=opt['rnn_dropout'], use_cuda=opt['cuda'])
            self.in_dim = opt['hidden_dim'] * 2
            self.rnn_drop = nn.Dropout(opt['rnn_dropout'])  # use on last layer output
        self.input_W_G = nn.Linear(self.in_dim, self.mem_dim)
        self.gcn = FirstGDABlock(self.mem_dim, opt['gcn_layers'], opt['rnn_layers'], 
                opt['gcn_dropout'], opt['rnn_dropout'])
        self.blocks = nn.ModuleList()
        # gcn layer
        for i in range(self.layers):
            self.blocks.append(GDABlock(opt['heads'], opt['hidden_dim'], opt['gcn_layers'], opt['rnn_layers'], 
                opt['gcn_dropout'], opt['rnn_dropout']))
        self.in_drop = nn.Dropout(opt['input_dropout'])
        # mlp output layer
        in_dim = opt['hidden_dim'] * 3
        layer = [nn.Linear(in_dim, opt['hidden_dim']), nn.ReLU()]
        for _ in range(self.opt['mlp_layers'] - 1):
            layer += [nn.Linear(opt['hidden_dim'], opt['hidden_dim']), nn.ReLU()]
        self.out_mlp = nn.Sequential(*layer)

        if self.opt['position_attn']:
            self.attn_layer = PositionAwareAttention(opt['hidden_dim'],
                    opt['hidden_dim'], 2*opt['pe_dim'], opt['attn_dim'])
            self.pe_emb = nn.Embedding(constant.MAX_LEN * 2 + 1, opt['pe_dim'])

        self.classifier = nn.Linear(opt['hidden_dim'], opt['num_class'])

        self.init_embeddings()

    def init_embeddings(self):

        if self.emb_matrix is None:
            self.emb.weight.data[1:, :].uniform_(-1.0, 1.0)
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)
        
        if self.opt['position_attn']:
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
        rnn_mask = masks
        gcn_mask = (words != constant.PAD_ID).unsqueeze(-2)
        
        word_embs = self.emb(words)
        embs = [word_embs]

        if self.opt['pos_dim'] > 0:
            embs += [self.pos_emb(pos)]
        if self.opt['ner_dim'] > 0:
            embs += [self.ner_emb(ner)]
        embs = torch.cat(embs, dim=2)
        embs = self.in_drop(embs)

        if self.opt.get('rnn', False):
            # embs = self.input_W_R(embs)
            gcn_inputs = self.rnn_drop(self.rnn(embs, masks)[0])
        else:
            gcn_inputs = embs
        gcn_inputs = self.input_W_G(gcn_inputs)

        def inputs_to_tree_reps(head, l):
            trees = [head_to_tree(head[i], l[i]) for i in range(len(l))]
            adj = [tree_to_adj(maxlen, tree, directed=False).reshape(1, maxlen, maxlen) for tree in trees]
            adj = np.concatenate(adj, axis=0)
            adj = torch.from_numpy(adj)
            return adj.cuda() if self.opt['cuda'] else adj

        adj = inputs_to_tree_reps(head.data, l)
        gcn_outputs, rnn_outputs, mask = self.gcn(adj, gcn_inputs, gcn_inputs, gcn_mask, rnn_mask)
     
        for i in range(self.layers):
            gcn_outputs, rnn_outputs = self.blocks[i](gcn_outputs, rnn_outputs, gcn_mask, rnn_mask)

        # attention
        if self.opt['position_attn']:
            # convert all negative PE numbers to positive indices
            # e.g., -2 -1 0 1 will be mapped to 98 99 100 101
            subj_pe_inputs = self.pe_emb(subj_pos + constant.MAX_LEN)
            obj_pe_inputs = self.pe_emb(obj_pos + constant.MAX_LEN)
            pe_features = torch.cat((subj_pe_inputs, obj_pe_inputs), dim=2)
            outputs = self.attn_layer(rnn_outputs, masks, gcn_outputs, pe_features)
        else:
            h = gcn_outputs + rnn_outputs
            h_out = pool(h, mask, "max")
            subj_mask, obj_mask = subj_pos.eq(0).eq(0).unsqueeze(2), obj_pos.eq(0).eq(0).unsqueeze(2)  # invert mask
            subj_out = pool(h, subj_mask, "max")
            obj_out = pool(h, obj_mask, "max")
            outputs = torch.cat([h_out, subj_out, obj_out], dim=1)
            outputs = self.out_mlp(outputs)

        outputs = self.classifier(outputs)
        return outputs


class BiLSTMBlock(nn.Module):
    def __init__(self, mem_dim, num_layers, rnn_dropout):
        super(BiLSTMBlock, self).__init__()
        self.mem_dim = mem_dim
        self.num_layers = num_layers
        self.rnn = MyRNN(self.mem_dim, self.mem_dim, num_layers=self.num_layers, 
            batch_first=True, dropout=rnn_dropout, bidirectional=True)
        self.linear = nn.Linear(self.mem_dim * 2, self.mem_dim)
    
    def forward(self, x, x_masks):
        rnn_output, ht = self.rnn(x, x_masks)
        rnn_output = self.linear(rnn_output)
        return rnn_output

class GDABlock(nn.Module):
    def __init__(self, heads, mem_dim, gcn_layers, rnn_layers, gcn_dropout, rnn_dropout):
        super(GDABlock, self).__init__()
        self.mem_dim = mem_dim
        self.rnn = BiLSTMBlock(self.mem_dim, rnn_layers, rnn_dropout)
        self.gcn = MultiDenseGCN(heads, self.mem_dim, gcn_layers, gcn_dropout)
        self.linear = nn.Linear(self.mem_dim, self.mem_dim)
        self.attn = MultiHeadAttention(heads, self.mem_dim)
    
    def forward(self, gcn_inputs, rnn_inputs, gcn_mask, rnn_mask, dropout=None):
        attn_tensor = self.attn(gcn_inputs, gcn_inputs, gcn_mask)
        attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]
        gcn_outputs = self.gcn(attn_adj_list, gcn_inputs)
        rnn_outputs = self.rnn(rnn_inputs, rnn_mask)
        # adj = attn_adj_list[0]
        # mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)
        attn = attention(gcn_outputs, rnn_outputs, gcn_mask, dropout)
        # gcn_outputs = torch.matmul(attn, gcn_outputs) 
        # rnn_outputs = torch.matmul(attn, rnn_outputs) 
        gcn_outputs = torch.matmul(attn, gcn_outputs) + gcn_inputs
        rnn_outputs = torch.matmul(attn, rnn_outputs) + rnn_inputs
        return gcn_outputs, rnn_outputs

class FirstGDABlock(nn.Module):
    def __init__(self, mem_dim, gcn_layers, rnn_layers, gcn_dropout, rnn_dropout):
        super(FirstGDABlock, self).__init__()
        self.mem_dim = mem_dim
        self.rnn = BiLSTMBlock(self.mem_dim, rnn_layers, rnn_dropout)
        self.gcn = DenseGCN(self.mem_dim, gcn_layers, gcn_dropout)
        self.linear = nn.Linear(self.mem_dim, self.mem_dim)
    
    def forward(self, adj, gcn_inputs, rnn_inputs, gcn_mask, rnn_mask, dropout=None):
        
        gcn_outputs = self.gcn(adj, gcn_inputs)
        rnn_outputs = self.rnn(rnn_inputs, rnn_mask)
        # adj = attn_adj_list[0]
        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)
        attn = attention(gcn_outputs, rnn_outputs, gcn_mask, dropout)
        # gcn_outputs = torch.matmul(attn, gcn_outputs) 
        # rnn_outputs = torch.matmul(attn, rnn_outputs) 
        gcn_outputs = torch.matmul(attn, gcn_outputs) + gcn_inputs
        rnn_outputs = torch.matmul(attn, rnn_outputs) + rnn_inputs
        return gcn_outputs, rnn_outputs, mask
