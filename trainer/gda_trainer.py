
import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np

from models import GDAClassifier
from trainer import unpack_batch, Trainer
from utils import torch_utils

class GDATrainer(Trainer):
    def __init__(self, opt, emb_matrix=None):
        self.opt = opt
        self.emb_matrix = emb_matrix
        self.model = GDAClassifier(opt, emb_matrix=emb_matrix)
        self.criterion = nn.CrossEntropyLoss()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]

        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])
    
    def update(self, batch):
        inputs, labels, tokens, head, subj_pos, obj_pos, lens = unpack_batch(batch, self.opt['cuda'])

        # Step 1 init and forward
        self.model.train()
        self.optimizer.zero_grad()

        logits = self.model(inputs)
        loss = self.criterion(logits, labels)
        loss_val = loss.item()

        # Step 2 backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])

        # Step 3 update
        self.optimizer.step()
        return loss_val 
    
    def predict(self, batch, unsort=True):
        inputs, labels, tokens, head, subj_pos, obj_pos, lens = unpack_batch(batch, self.opt['cuda'])
        orig_idx = batch[11]
        
        self.model.eval()

        logits = self.model(inputs)
        loss = self.criterion(logits, labels)
        loss_val = loss.item()

        probs = F.softmax(logits, 1).data.cpu().numpy().tolist()
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
        if unsort:
            _, predictions, probs = [list(t) for t in zip(*sorted(zip(orig_idx, predictions, probs)))]
        
        return predictions, probs, loss_val