'''
Wrapper of RNN Layer.
'''
import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from utils import constant
class MyRNN(nn.Module):
    """ A wrapper of rnn layer."""
    def __init__(self,
                input_size, 
                hidden_dim, 
                num_layers=1, 
                bidirectional=False, 
                dropout=0, 
                batch_first=True, 
                use_cuda=True,
                rnn_type='lstm'): # rnn, lstm, gru
        super(MyRNN, self).__init__()

        self.use_cuda = use_cuda
        self.hidden_dim = hidden_dim
        self.input_size = input_size
        self.num_layers = num_layers
        self.direct = 2 if bidirectional else 1
        self.batch_first = batch_first
    
        rnn_type = rnn_type.upper()
        args = dict(input_size=input_size, hidden_size=hidden_dim, 
                        num_layers=num_layers, bidirectional=bidirectional, 
                        dropout=dropout, batch_first=batch_first)
        self.rnn = getattr(nn, rnn_type)(**args)

    def rnn_zero_state(self, batch_size):
        total_layers = self.num_layers * self.direct
        state_shape = (total_layers, batch_size, self.hidden_dim)
        h0 = c0 = torch.zeros(*state_shape, requires_grad=False)
        if self.use_cuda:
            return h0.cuda(), c0.cuda()
        else:
            return h0, c0        
    
    def forward(self, inputs, masks=None):
        batch_size = inputs.size(0)
        h0, c0 = self.rnn_zero_state(batch_size)
        if masks is None:
            rnn_outputs, (ht, ct) = self.rnn(inputs, (h0, c0))
        else:
            seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())
            rnn_inputs = pack_padded_sequence(inputs, seq_lens, batch_first=self.batch_first)
            rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
            rnn_outputs, _ = pad_packed_sequence(rnn_outputs, batch_first=self.batch_first)
        
        return rnn_outputs, ht
