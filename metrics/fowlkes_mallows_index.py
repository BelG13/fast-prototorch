import torch
from math import sqrt
from .precision import Precision
from .true_positive_rate import TPR
from ._BaseMetric import _BaseMetric


class FM(_BaseMetric):
    def metric_func(self, logits: torch.Tensor, y: torch.Tensor, default: float = None):
        """Compute he fowlkes_mallows_index

        Args:
            logits (torch.Tensor): output from the model (B, C)
            y (torch.Tensor): true labels (B, 1)
            default (float): Value to return if the metric is not well defined

        Returns:
            float: fowlkes_mallows_index of that batch
        """

        tpr = TPR()(logits, y)
        ppv = Precision()(logits, y)

        return sqrt(tpr * ppv)
