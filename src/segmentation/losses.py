import torch
from torch.nn import Module


class ComboLoss(Module):
    def __init__(
            self,
            alpha: float = 0.5, # < 0.5 penalises FP more, > 0.5 penalises FN more
            beta: float = 0.1,
        ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = 1e-7

    def forward(self, inputs, targets, smooth=1):
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        #True Positives, False Positives & False Negatives
        intersection = (inputs * targets).sum()    
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
        # inputs = torch.clamp(inputs,  self.eps, 1.0 - self.eps)
        out = - (self.beta * (
            (targets * torch.log(inputs + self.eps)) + ((1 - self.beta) * (1.0 - targets) * torch.log(1.0 - inputs + self.eps))))
        weighted_ce = out.mean(-1)
        combo = (self.alpha * weighted_ce) - ((1 - self.alpha) * dice)
        
        return combo
    