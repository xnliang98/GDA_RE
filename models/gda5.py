import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.layers import pool, MyRNN, DenseGCN, MultiDenseGCN, GCNLayer
from models.layers import MultiHeadAttention, attention
from models.layers import Tree, head_to_tree, tree_to_adj
from utils import constant, torch_utils

class GDAClassifier(nn.Module):
    def __init__(self, opt, emb_matrix=None):
        super(GDAClassifier, self).__init__()
        self.opt = opt
        self.emb_matrix = emb_matrix
        self.layers = opt['layers']
        self.mem_dim = opt['hidden_dim']
        # create embedding layers
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt['pos_dim']) if opt['pos_dim'] > 0 else None
        self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), opt['ner_dim']) if opt['ner_dim'] > 0 else None
        self.subj_emb = nn.Embedding(constant.MAX_LEN * 2 + 1, 30)
        self.obj_emb = nn.Embedding(constant.MAX_LEN * 2 + 1, 30)
        self.tree_emb = nn.Embedding(len(constant.DEPREL_TO_ID), 30)

        self.in_dim = opt['emb_dim'] + opt['pos_dim'] + opt['ner_dim'] + 30
        self.in_without_rnn = nn.Linear(self.in_dim, self.mem_dim * 2)
        
        input_size = self.in_dim
        self.rnn = MyRNN(input_size, opt['hidden_dim'], 2,
                bidirectional=True, dropout=opt['rnn_dropout'], use_cuda=opt['cuda'])
        self.in_dim = opt['hidden_dim']
        self.rnn_drop = nn.Dropout(opt['rnn_dropout'])  # use on last layer output

        self.gcn1 = DenseGCN(opt['hidden_dim'], 3, opt['gcn_dropout'])
        self.gcn2 = DenseGCN(opt['hidden_dim'], 3, opt['gcn_dropout'])
        # self.gcn1 = GCNLayer(opt['hidden_dim'], opt['hidden_dim'], 4, opt['gcn_dropout'])
        # self.gcn2 = GCNLayer(opt['hidden_dim'], opt['hidden_dim'], 4, opt['gcn_dropout'])
        self.in_drop = nn.Dropout(opt['input_dropout'])
        self.dropout = nn.Dropout(opt['gcn_dropout'])

        self.head_lstm = HEADLSTM(opt['hidden_dim'], opt['rnn_layers'], opt['rnn_dropout'])
        self.tail_lstm = TAILLSTM(opt['hidden_dim'], opt['rnn_layers'], opt['rnn_dropout'])
        # self.out_lstm = HEADLSTM(opt['hidden_dim'], opt['rnn_layers'], opt['rnn_dropout'])
        self.linear1 = nn.Linear(opt['hidden_dim'] * 4 + 30, opt['hidden_dim'] * 2)
        self.linear2 = nn.Linear(opt['hidden_dim'] * 4 + 30, opt['hidden_dim'] * 2)
        self.linear3 = nn.Linear(opt['hidden_dim'] * 2, opt['hidden_dim'] * 1)
        self.out = nn.Linear(opt['hidden_dim'] * 3, opt['hidden_dim'])
        self.classifier = nn.Linear(opt['hidden_dim'], opt['num_class'])

        self.init_embeddings()

    def init_embeddings(self):
        # nn.init.xavier_uniform_(self.linear1.weight)
        # nn.init.xavier_uniform_(self.linear2.weight)
        # nn.init.xavier_uniform_(self.linear3.weight)
        # nn.init.xavier_uniform_(self.classifier.weight)
        # nn.init.xavier_uniform_(self.out.weight)
        self.subj_emb.weight.data[1:, :].uniform_(-1.0, 1.0)
        self.obj_emb.weight.data[1:, :].uniform_(-1.0, 1.0)
        self.tree_emb.weight.data[1:, :].uniform_(-1.0, 1.0)
        if self.emb_matrix is None:
            self.emb.weight.data[1:, :].uniform_(-1.0, 1.0)
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)
        
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
        embs += [self.tree_emb(deprel)]
        embs = torch.cat(embs, dim=2)
        embs = self.in_drop(embs)

        
        inputs = self.rnn_drop(self.rnn(embs, masks)[0])

        def inputs_to_tree_reps(head, l):
            trees = [head_to_tree(head[i], l[i]) for i in range(len(l))]
            adj = [tree_to_adj(maxlen, tree, directed=False).reshape(1, maxlen, maxlen) for tree in trees]
            adj = np.concatenate(adj, axis=0)
            adj = torch.from_numpy(adj)
            return adj.cuda() if self.opt['cuda'] else adj
        adj = inputs_to_tree_reps(head.data, l)
        
        gcn_out1 = self.gcn1(adj, inputs[:, :, :self.mem_dim])
        gcn_out2 = self.gcn2(adj, inputs[:, :, self.mem_dim:])

        gcn_out = torch.cat([gcn_out2, gcn_out1], dim=-1)
        assert gcn_out.shape == inputs.shape
        subj_inputs = self.subj_emb(subj_pos + constant.MAX_LEN)
        obj_inputs = self.obj_emb(obj_pos + constant.MAX_LEN)
        inputs = torch.cat([gcn_out, inputs, subj_inputs], dim=-1)
        inputs = self.linear1(inputs)
        inputs = self.dropout(inputs)

        head_out = self.head_lstm(inputs, masks)
        tail_inputs = torch.cat([head_out, gcn_out, obj_inputs], dim=-1)
        tail_inputs = self.dropout(self.linear2(tail_inputs))
        tail_out = self.tail_lstm(tail_inputs, masks)

        outputs = self.linear3(tail_out)
        outputs = self.dropout(outputs)
        h = outputs
        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)
        h_out = pool(h, mask, "max")
        subj_mask, obj_mask = subj_pos.eq(0).eq(0).unsqueeze(2), obj_pos.eq(0).eq(0).unsqueeze(2)  # invert mask
        subj_out = pool(h, subj_mask, "max")
        obj_out = pool(h, obj_mask, "max")
        outputs = torch.cat([h_out, subj_out, obj_out], dim=1)

        outputs = self.out(outputs)
        outputs = F.relu(outputs)
        outputs = self.classifier(outputs)
        return outputs


class HEADLSTM(nn.Module):
    def __init__(self, mem_dim, num_layers, rnn_dropout):
        super(HEADLSTM, self).__init__()
        self.mem_dim = mem_dim
        self.in_dim = mem_dim * 2
        self.num_layers = num_layers
        self.rnn = MyRNN(self.in_dim, self.mem_dim, num_layers=self.num_layers, 
            batch_first=True, dropout=rnn_dropout, bidirectional=True)
    
    def forward(self, x, x_masks):
        rnn_output, ht = self.rnn(x, x_masks)
        return rnn_output

class TAILLSTM(nn.Module):
    def __init__(self, mem_dim, num_layers, rnn_dropout):
        super(TAILLSTM, self).__init__()
        self.mem_dim = mem_dim
        self.in_dim = mem_dim * 2
        self.num_layers = num_layers
        self.rnn = MyRNN(self.in_dim, self.mem_dim, num_layers=self.num_layers, 
            batch_first=True, dropout=rnn_dropout, bidirectional=True)
    
    def forward(self, x, x_masks):
        rnn_output, ht = self.rnn(x, x_masks)
        return rnn_output



