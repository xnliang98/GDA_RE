import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.nn import init

from models.layers import MyRNN, PositionAwareAttention
from utils import constant, torch_utils

class PALSTM(nn.Module):
    """ A classifier use lstm and position aware attention 
        to extract relations.
    """
    def __init__(self, opt, emb_matrix=None):
        super(PALSTM, self).__init__()
        