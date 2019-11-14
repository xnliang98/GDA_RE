
import torch
import torch.nn as nn
import torch.nn.functional as F

def pool(h, mask, pool_type='max'):
    if pool_type == 'max':
        h = h.masked_fill(mask, -1e9)
        return torch.max(h, 1)[0]
    elif pool_type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)

class SingleGCNLayer(nn.Module):
    def __init__(self, in_dim, mem_dim):
        super(SingleGCNLayer, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.linear = nn.Linear(self.in_dim, self.mem_dim)
    
    def conv_l2(self):
        return [self.linear.weight, self.linear.bias]
    
    def forward(self, adj, inputs):
        denom = adj.sum(2).unsqueeze(2) + 1
        Ax = adj.bmm(inputs)
        AxW = self.linear(Ax)
        AxW = AxW + self.linear(inputs)
        AxW /= denom
        gAxW = F.relu(AxW)

        return gAxW

class GCNLayer(nn.Module):
    def __init__(self, in_dim, mem_dim, num_layers, gcn_dropout, no_adj=False):
        super(GCNLayer, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.num_layers = num_layers
        self.gcn_drop = nn.Dropout(gcn_dropout)
        
        self.gcn = nn.ModuleList()
        for layer in range(self.num_layers):
            input_dim = self.in_dim if layer == 0 else self.mem_dim
            self.gcn.append(SingleGCNLayer(input_dim, self.mem_dim))

    def conv_l2(self):
        conv_weights = []
        for layer in self.gcn:
            conv_weights += layer.conv_l2()
        return sum([x.pow(2).sum() for x in conv_weights])

    def forward(self, adj, inputs):
        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)

        for layer in range(self.num_layers):
            inputs = self.gcn[layer](adj, inputs)
            if layer < self.num_layers - 1:
                inputs = self.gcn_drop(inputs)

        return inputs, mask