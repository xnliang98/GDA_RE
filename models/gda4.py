"""
AGGCNs model for relation extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.layers import pool, MyRNN, DenseGCN, MultiDenseGCN
from models.layers import MultiHeadAttention
from models.layers import Tree, head_to_tree, tree_to_adj
from utils import constant, torch_utils


class GDAClassifier(nn.Module):
    """ A wrapper classifier for GCNRelationModel. """

    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        self.gcn_model = GCNRelationModel(opt, emb_matrix=emb_matrix)
        in_dim = opt['hidden_dim']
        self.classifier = nn.Linear(in_dim, opt['num_class'])
        self.opt = opt

    def forward(self, inputs):
        outputs, pooling_output = self.gcn_model(inputs)
        logits = self.classifier(outputs)
        return logits, pooling_output

class GCNRelationModel(nn.Module):
    def __init__(self, opt, emb_matrix=None):
        super(GCNRelationModel, self).__init__()
        self.opt = opt
        self.emb_matrix = emb_matrix

        # create embedding layers
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt['pos_dim']) if opt['pos_dim'] > 0 else None
        self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), opt['ner_dim']) if opt['ner_dim'] > 0 else None
        embeddings = (self.emb, self.pos_emb, self.ner_emb)
        self.init_embeddings()

        # gcn layer
        self.gcn = AGGCN(opt, embeddings)

        # mlp output layer
        in_dim = opt['hidden_dim'] * 3
        layers = [nn.Linear(in_dim, opt['hidden_dim']), nn.ReLU()]
        for _ in range(self.opt['mlp_layers'] - 1):
            layers += [nn.Linear(opt['hidden_dim'], opt['hidden_dim']), nn.ReLU()]
        self.out_mlp = nn.Sequential(*layers)

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

        def inputs_to_tree_reps(head, l):
            trees = [head_to_tree(head[i], l[i]) for i in range(len(l))]
            adj = [tree_to_adj(maxlen, tree, directed=False).reshape(1, maxlen, maxlen) for tree in trees]
            adj = np.concatenate(adj, axis=0)
            adj = torch.from_numpy(adj)
            return adj.cuda() if self.opt['cuda'] else adj

        adj = inputs_to_tree_reps(head.data, l)
        h, pool_mask = self.gcn(adj, inputs)

        # pooling
        subj_mask, obj_mask = subj_pos.eq(0).eq(0).unsqueeze(2), obj_pos.eq(0).eq(0).unsqueeze(2)  # invert mask
        pool_type = "max"
        h_out = pool(h, pool_mask, pool_type)
        subj_out = pool(h, subj_mask, pool_type)
        obj_out = pool(h, obj_mask, pool_type)
        outputs = torch.cat([h_out, subj_out, obj_out], dim=1)
        outputs = self.out_mlp(outputs)

        return outputs, h_out

class AGGCN(nn.Module):
    def __init__(self, opt, embeddings):
        super(AGGCN, self).__init__()
        self.opt = opt
        
        self.emb, self.pos_emb, self.ner_emb = embeddings
        self.use_cuda = opt['cuda']
        self.mem_dim = opt['hidden_dim']

        self.pe_obj_emb = nn.Embedding(constant.MAX_LEN * 2 + 1, opt['pe_emb'])
        self.pe_subj_emb = nn.Embedding(constant.MAX_LEN * 2 + 1, opt['pe_emb'])
        self.in_dim = opt['emb_dim'] + opt['pos_dim'] + opt['ner_dim'] 

        if opt.get('rnn', False):
            input_size = self.in_dim 
            self.rnn = MyRNN(input_size, opt['hidden_dim'], opt['rnn_layers'],
                bidirectional=True, dropout=opt['rnn_dropout'], use_cuda=opt['cuda'])
            self.in_dim = opt['hidden_dim'] * 4
            self.rnn_drop = nn.Dropout(opt['rnn_dropout'])  # use on last layer output
        self.input_W_G = nn.Linear(self.in_dim, self.mem_dim)

        self.in_drop = nn.Dropout(opt['input_dropout'])
        self.num_layers = opt['num_layers']

        self.layers_head = nn.ModuleList()
        self.layers_tail = nn.ModuleList()
        self.heads = opt['heads']
        self.sublayer_head = opt['sublayer_first']
        self.sublayer_tail = opt['sublayer_second']
        self.layers_h2t = nn.ModuleList()
        self.layers_i2h = nn.ModuleList()
        self.layers_dropout = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                self.layers_head.append(DenseGCN(self.mem_dim, self.sublayer_head, opt['gcn_dropout']))
                self.layers_tail.append(DenseGCN(self.mem_dim, self.sublayer_tail, opt['gcn_dropout']))
            else:
                self.layers_head.append(MultiDenseGCN(self.heads, self.mem_dim, self.sublayer_head, opt['gcn_dropout']))
                self.layers_tail.append(MultiDenseGCN(self.heads, self.mem_dim, self.sublayer_tail, opt['gcn_dropout']))
            self.layers_i2h.append(nn.Linear(self.mem_dim + opt['pe_emb'], self.mem_dim))
            self.layers_h2t.append(nn.Linear(self.mem_dim * 2 + opt['pe_emb'], self.mem_dim))
            self.layers_dropout.append(nn.Dropout(0.5))
        
        self.aggregate_W = nn.Linear(len(self.layers_h2t) * self.mem_dim, self.mem_dim)
        self.attn = MultiHeadAttention(self.heads, self.mem_dim)
        self.init_weights()
    
    def init_weights(self):
        self.pe_obj_emb.weight.data.uniform_(-1.0, 1.0)
        self.pe_subj_emb.weight.data.uniform_(-1.0, 1.0)
        for i in range(self.num_layers):
            nn.init.xavier_uniform_(self.layers_h2t[i].weight)
            nn.init.xavier_uniform_(self.layers_i2h[i].weight)
        
        # nn.init.xavier_uniform_(self.aggregate_W.weight)
        # nn.init.xavier_uniform_(self.input_W_G.weight)
    
    def forward(self, adj, inputs):
        words, masks, pos, ner, deprel, head, subj_pos, obj_pos, subj_type, obj_type = inputs # unpack
        src_mask = (words != constant.PAD_ID).unsqueeze(-2)
        word_embs = self.emb(words)
        embs = [word_embs]

        if self.opt['pos_dim'] > 0:
            embs += [self.pos_emb(pos)]
        if self.opt['ner_dim'] > 0:
            embs += [self.ner_emb(ner)]
        subj_pe_inputs = self.pe_obj_emb(subj_pos + constant.MAX_LEN)
        obj_pe_inputs = self.pe_subj_emb(obj_pos + constant.MAX_LEN)
        # embs += [subj_pe_inputs, obj_pe_inputs]
        embs = torch.cat(embs, dim=-1)
        embs = self.in_drop(embs)

        rnn_outputs, hidden = self.rnn(embs, masks)
        shape_t = rnn_outputs.size()
        h = torch.cat([hidden[1], hidden[0]], dim=-1).unsqueeze(1).expand(shape_t)

        head_inputs = self.input_W_G(torch.cat([rnn_outputs, h], dim=-1))

        layer_list = []
        head_outputs = head_inputs
        tail_outputs = head_inputs
        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)
        for i in range(len(self.layers_h2t)):
            if i < 1:
                head_outputs = F.relu(self.layers_i2h[i](torch.cat([head_outputs, obj_pe_inputs], dim=-1)))
                head_outputs = self.layers_head[i](adj, head_outputs)
                tmp = torch.cat([head_outputs, tail_outputs, subj_pe_inputs], dim=-1)
                tmp = self.layers_dropout[i](self.layers_h2t[i](tmp))

                tail_outputs = self.layers_tail[i](adj, tmp)
                layer_list.append(tail_outputs)
            else:
                head_outputs = F.relu(self.layers_i2h[i](torch.cat([head_outputs, obj_pe_inputs], dim=-1)))
                attn_tensor = self.attn(head_outputs, head_outputs, src_mask)
                attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]
                head_outputs = self.layers_head[i](attn_adj_list, head_outputs)

                attn_tensor = self.attn(tail_outputs, tail_outputs, src_mask)
                attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]
                tmp = torch.cat([head_outputs, tail_outputs, subj_pe_inputs], dim=-1)
                tmp = self.layers_dropout[i](self.layers_h2t[i](tmp))
                tail_outputs = self.layers_tail[i](attn_adj_list, tmp)
                layer_list.append(tail_outputs)

        aggregate_out = torch.cat(layer_list, dim=2)
        dcgcn_output = self.aggregate_W(aggregate_out)

        return dcgcn_output, mask

