import torch.nn as nn

class MultiTaskLoss(nn.Module):
    def __init__(self, reg_loss, alpha=1.0, beta=1.0):
        super(MultiTaskLoss, self).__init__()
        self.reg_loss = reg_loss
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits, labels):
        reg_logit = logits[:, 0]
        bce_logit = logits[:, 1]
        
        reg_label = labels[:, 0]
        bce_label = labels[:, 1]
        
        reg_loss = self.reg_loss(reg_logit, reg_label)
        bce_loss = self.bce_loss(bce_logit, bce_label)
        
        tot_loss = self.alpha * reg_loss + self.beta * bce_loss
        
        return tot_loss