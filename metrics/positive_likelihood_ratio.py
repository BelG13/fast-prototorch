import torch
from .true_positive_rate import TPR
from .false_positive_rate import FPR
from ._BaseMetric import _BaseMetric


class LRPlus(_BaseMetric):

    def metric_func(self, logits: torch.Tensor, y: torch.Tensor, default: float = 0):
        """Compute the positive likelihood ratio

        Args:
            logits (torch.Tensor): output from the model (B, C)
            y (torch.Tensor): true labels (B, 1)
            default (float): Value to return if the metric is not well defined

        Returns:
            float: positive likelihood ratio of that batch
        """

        tpr = TPR()(logits, y)
        fpr = FPR()(logits, y)

        return tpr / fpr if fpr != 0 else default
