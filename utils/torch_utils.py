"""
Utility functions for torch.
"""

import torch
from torch import nn, optim
from torch.optim import Optimizer

class MyAdagrad(Optimizer):
    """My modification of the Adagrad optimizer that allows to specify an initial
    accumulater value. This mimics the behavior of the default Adagrad implementation 
    in Tensorflow. The default PyTorch Adagrad uses 0 for initial acculmulator value.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        lr_decay (float, optional): learning rate decay (default: 0)
        init_accu_value (float, optional): initial accumulater value.
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, params, lr=1e-2, lr_decay=0, init_accu_value=0.1, weight_decay=0):
        defaults = dict(lr=lr, lr_decay=lr_decay, init_accu_value=init_accu_value, \
                weight_decay=weight_decay)
        super(MyAdagrad, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['sum'] = torch.ones(p.data.size()).type_as(p.data) *\
                        init_accu_value

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['sum'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                state['step'] += 1

                if group['weight_decay'] != 0:
                    if p.grad.data.is_sparse:
                        raise RuntimeError("weight_decay option is not compatible with sparse gradients ")
                    grad = grad.add(group['weight_decay'], p.data)

                clr = group['lr'] / (1 + (state['step'] - 1) * group['lr_decay'])

                if p.grad.data.is_sparse:
                    grad = grad.coalesce()  # the update is non-linear so indices must be unique
                    grad_indices = grad._indices()
                    grad_values = grad._values()
                    size = torch.Size([x for x in grad.size()])

                    def make_sparse(values):
                        constructor = type(p.grad.data)
                        if grad_indices.dim() == 0 or values.dim() == 0:
                            return constructor()
                        return constructor(grad_indices, values, size)
                    state['sum'].add_(make_sparse(grad_values.pow(2)))
                    std = state['sum']._sparse_mask(grad)
                    std_values = std._values().sqrt_().add_(1e-10)
                    p.data.add_(-clr, make_sparse(grad_values / std_values))
                else:
                    state['sum'].addcmul_(1, grad, grad)
                    std = state['sum'].sqrt().add_(1e-10)
                    p.data.addcdiv_(-clr, grad, std)

        return loss

def get_optimizer(name, parameters, lr):
    ''' Choose optimizer with name and return it.
        Args:
            name (str): name of optimizer in 
                ['sgd', 'adagrad', 'myadagrad', 'adam', 'adamax']
            parameters (model.parameters()): model's parameters.
            lr (float): learning rate.
        return optimizer
    '''
    if name == 'sgd':
        return optim.SGD(parameters, lr=lr)
    elif name in ['adagrad', 'myadagrad']:
        return MyAdagrad(parameters, lr=lr, init_accu_value=0.1)
    elif name == 'adam':
        return optim.Adam(parameters, betas=(0.9, 0.99)) # use default lr
    elif name == 'adamax':
        return optim.adamax(parameters) # use default lr
    else:
        raise Exception("Unsupport optimizer: {}".format(name))

def change_lr(optimizer, new_lr):
    ''' Change learning rate of optimzer.
        Args:
            optimizer (torch.optim): optimizer.
            new_lr (float): new learning rate.
    '''
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def flatten_indices(seq_lens, width):
    flat = []
    for i, l in enumerate(seq_lens):
        for j in range(l):
            flat.append(i * width + j)
    return flat

def set_cuda(var, cuda):
    ''' If use cuda, change var to cuda variables and return.
        Args:
            var (torch.tensor): variable.
            cuda (boolean): use cuda or not use cuda.
    '''
    if cuda:
        return var.cuda()
    return var

def keep_partial_grad(grad, topk):
    ''' Keep only the topk rows of grads.
        Args:
            grad (torch.tensor): grad of model.
            topk (int): number to keep.
    '''
    assert topk < grad.size(0)
    grad.data.[topk:].zero_()
    return grad

# Model io
def save(model, optimizer, opt, filename):
    ''' Save model, optimizer, opt to filename
        Args:
            model (nn.Module): model.
            optimizer (torch.optim): optimizer.
            opt (dict): option params, config.
            filename (str): filename to save.
    '''
    params = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': opt
    }
    try:
        torch.save(params, filename)
    except BaseException:
        print("[ Warning: model saving failed. ]")
    

def load(model, optimizer, filename):
    ''' Load model and optimizer from filename.
        Args:
            model (nn.Module): model.
            optimizer (torch.optim): optimizer.
            filename (str): filename to load model and optimizer.
        returns:
            model, optimizer, opt
    '''
    try:
        dump = torch.load(filename)
    except BaseException:
        print("[ Fail: model loading failed. ]")
    if model is not None:
        model.load_state_dict(dump['model'])
    if optimizer is not None:
        optimizer.load_state_dict(dump['optimzier'])
    opt = dump['config']
    return model, optimizer, opt

def load_config(filename):
    ''' Load config of model from filename.
        Args:
            filename (str): filename to load config.
        returns:
            opt (config)
    '''
    try:
        dump = torch.load(filename)
    except BaseException:
        print("[ Fail: model loading failed. ]")
    opt = dump['config']
    return opt

import copy 

def clones(module, N):
    """ Produce N identical layers.
        Args:
            module (nn.Module): layers need to be copy.
            N (int): copy times.
        returns:
            nn.ModuleList of N identical module.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])