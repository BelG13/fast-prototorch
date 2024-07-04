import torch
from ._BaseMetric import _BaseMetric
from .positive_likelihood_ratio import LRPlus
from .negative_likelihood_ratio import LRMinus

class DOR(_BaseMetric):

    def metric_func(self, logits: torch.Tensor, y: torch.Tensor, default: float = 0.0):
        """Compute the diagnostic odds ratio

        Args:
            logits (torch.Tensor): output from the model (B, C)
            y (torch.Tensor): true labels (B, 1)
            default (float): Value to return if the metric is not well defined
        
        Returns:
            float: diagnostic odds ratio of that batch
        """
        
        lr_plus = LRPlus()(logits, y)
        lr_minus = LRMinus()(logits, y)
        
        return lr_plus / lr_minus if lr_minus != 0 else default