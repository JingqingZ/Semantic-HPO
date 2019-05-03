
import torch
from torch import nn
import torch.nn.functional as F

# TODO: apply mask
class Loss():
    def __init__(self):
        # self.cross_entropy = nn.CrossEntropyLoss(
        #     reduction='mean'
        # )
        pass

    def resconstruction(self, logits, target_ids, mask=None):
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_ids.reshape(-1)
        )
        return loss

    def alpha_cross_entropy(self, logits, target_logits, mask=None):
        loss = F.binary_cross_entropy_with_logits(
            logits,
            target_logits
        )
        return loss
