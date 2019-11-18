
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
    def __init__(self, in_dim, mem_dim, num_layers, gcn_dropout):
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

class DenseGCN(nn.Module):
    def __init__(self, mem_dim, num_layers, gcn_dropout):
        super(DenseGCN, self).__init__()
        
        self.mem_dim = mem_dim
        self.num_layers = num_layers
        self.gcn_drop = nn.Dropout(gcn_dropout)
        self.head_dim = self.mem_dim // self.num_layers
        
        # linear transformation
        self.linear_output = nn.Linear(self.mem_dim, self.mem_dim)

        self.gcn = nn.ModuleList()
        for layer in range(self.num_layers):
            self.gcn.append(SingleGCNLayer(self.mem_dim + self.head_dim * layer, self.head_dim))

    def forward(self, adj, inputs, is_transform=True):
        # mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)
        outputs = inputs
        cache_list = [outputs]
        output_list = []

        for layer in range(self.num_layers):
            single_output = self.gcn[layer](adj, outputs)

            cache_list.append(single_output)
            outputs = torch.cat(cache_list, dim=2)
            output_list.append(self.gcn_drop(single_output))
        
        gcn_outputs = torch.cat(output_list, dim=2)
        # print(gcn_outputs.shape)
        # gcn_outputs = gcn_outputs + inputs
        if is_transform:
            out = self.linear_output(gcn_outputs)
        else:
            out = gcn_outputs
        return out

class MultiDenseGCN(nn.Module):
    def __init__(self, heads, mem_dim, num_layers, gcn_dropout):
        super(MultiDenseGCN, self).__init__()

        self.heads = heads
        self.mem_dim = mem_dim
        self.num_layers = num_layers

        self.linear = nn.Linear(self.mem_dim * self.heads, self.mem_dim)
        
        self.gcns = nn.ModuleList()
        for i in range(self.heads):
            self.gcns.append(DenseGCN(self.mem_dim, self.num_layers, gcn_dropout))
    
    def forward(self, adj, gcn_inputs):
        multi_head_list = []

        for i in range(self.heads):
            gcn_outputs = self.gcns[i](adj[i], gcn_inputs, False)
            multi_head_list.append(gcn_outputs)
        
        final_output = torch.cat(multi_head_list, dim=2)
        out = self.linear(final_output)

        return out

if __name__ == "__main__":
    adj = torch.rand(64, 100, 100)
    inputs = torch.randn(64, 100, 200)
    model = MultiDenseGCN(4, 200, 4, 0.5)
    model(adj, inputs)