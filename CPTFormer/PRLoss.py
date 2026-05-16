import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicPairedRobustnessLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits_o, logits_c, labels):
        loss_o = self.base_criterion(logits_o, labels)
        loss_c = self.base_criterion(logits_c, labels)
        with torch.no_grad():
            omega = torch.exp(loss_c)
        paired_loss = (omega * (loss_o + loss_c)).mean()
        
        return paired_loss