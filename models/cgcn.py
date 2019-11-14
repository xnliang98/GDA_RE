"""
C-GCN model for relation extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.layers import GCNLayer, pool, MyRNN
from models.layers import Tree, head_to_tree, tree_to_adj
from utils import constant, torch_utils

class CGCNClassifier(nn.Module):
    def __init__(self, opt, emb_matrix=None):
        super(CGCNClassifier, self).__init__()
        self.opt = opt
        self.emb_matrix = emb_matrix

        # create embedding layers
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt['pos_dim']) if opt['pos_dim'] > 0 else None
        self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), opt['ner_dim']) if opt['ner_dim'] > 0 else None
        self.in_drop = nn.Dropout(opt['input_dropout'])
        embeddings = (self.emb, self.pos_emb, self.ner_emb)
        self.init_embeddings()
        
        # GCN layers
        self.in_dim = opt['emb_dim'] + opt['pos_dim'] + opt['ner_dim']
        self.mem_dim = opt['hidden_dim']

        if opt.get('rnn', False):
            input_size = self.in_dim
            self.rnn = MyRNN(input_size, opt['hidden_dim'], opt['rnn_layers'],
                bidirectional=True, dropout=opt['rnn_dropout'], use_cuda=opt['cuda'])
            self.in_dim = opt['hidden_dim'] * 2
            self.rnn_drop = nn.Dropout(opt['rnn_dropout'])  # use on last layer output
        self.input_W_G = nn.Linear(self.in_dim, self.mem_dim)
        self.gcn = GCNLayer(self.in_dim, self.mem_dim, self.num_layers, 
            opt['input_dropout'], opt.get('no_adj', False))
        in_dim = opt['hidden_dim'] * 3
        layers = [nn.Linear(in_dim, opt['hidden_dim']), nn.ReLU()]
        for _ in range(opt['mlp_layers'] - 1):
            layers += [nn.Linear(opt['hidden_dim'], opt['hidden_dim']), nn.ReLU()]
        self.out_mlp = nn.Sequential(*layers)
        self.classifier = nn.Linear(opt['hidden_dim'], opt['num_class'])

    def init_embeddings(self):
        if self.emb_matrix is None:
            nn.init.uniform_(self.emb.weight.data[1:, :])
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)
        # decide finetuning
        if self.opt['topn'] <= 0:
            print("Do not finetune word embedding layer.")
            self.emb.weight.requires_grad = False
        elif self.opt['topn'] < self.opt['vocab_size']:
            print("Finetune top {} word embeddings.".format(self.opt['topn']))
            self.emb.weight.register_hook(lambda x: \
                    torch_utils.keep_partial_grad(x, self.opt['topn']))
        else:
            print("Finetune all embeddings.")
    
    def forward(self, inputs):
        words, masks, pos, ner, deprel, head, subj_pos, obj_pos, subj_type, obj_type = inputs # unpack
        seq_length = (masks.data.cpu().numpy() == 0).astype(np.int64).sum(1)
        maxlen = max(seq_length)

        def inputs_to_tree_reps(head, words, l, prune, subj_pos, obj_pos):
            head, words, subj_pos, obj_pos = head.cpu().numpy(), words.cpu().numpy(), subj_pos.cpu().numpy(), obj_pos.cpu().numpy()
            trees = [head_to_tree(head[i], words[i], l[i], prune, subj_pos[i], obj_pos[i]) for i in range(len(l))]
            adj = [tree_to_adj(maxlen, tree, directed=False, self_loop=False).reshape(1, maxlen, maxlen) for tree in trees]
            adj = np.concatenate(adj, axis=0)
            adj = torch.from_numpy(adj)
            return adj.cuda() if self.opt['cuda'] else adj
        
        adj = inputs_to_tree_reps(head.data, words.data, seq_length, self.opt['prune_k'], subj_pos.data, obj_pos.data)
        word_embs = self.emb(words)
        embs = [word_embs]
        if self.opt['pos_dim'] > 0:
            embs += [self.pos_emb(pos)]
        if self.opt['ner_dim'] > 0:
            embs += [self.ner_emb(ner)]
        embs = torch.cat(embs, dim=2)
        embs = self.in_drop(embs)

        # rnn layer
        if self.opt.get('rnn', False):
            gcn_inputs = self.rnn_drop(self.rnn(embs, masks)[0])
        else:
            gcn_inputs = embs
        
        # gcn layers
        h, pool_mask = self.gcn(adj, gcn_inputs)

        # pooling
        subj_mask, obj_mask = subj_pos.eq(0).eq(0).unsqueeze(2), obj_pos.eq(0).eq(0).unsqueeze(2) # invert mask
        pool_type = self.opt['pooling']
        h_out = pool(h, pool_mask, pool_type=pool_type)
        subj_out = pool(h, subj_mask, pool_type=pool_type)
        obj_out = pool(h, obj_mask, pool_type=pool_type)
        outputs = torch.cat([h_out, subj_out, obj_out], dim=1)
        outputs = self.out_mlp(outputs)
        outputs = self.classifier(outputs)

        return outputs, h_out
    
    def conv_l2(self):
        return self.gcn.conv_l2()