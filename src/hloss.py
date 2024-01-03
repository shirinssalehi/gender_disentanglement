import torch
from configs_disentanglement import EPSILON

def entropy_loss(preds):
    """
    Returns the entropy loss: negative of the entropy present in the
    input distribution
    """
    return torch.mean(preds * torch.log(preds + EPSILON))
