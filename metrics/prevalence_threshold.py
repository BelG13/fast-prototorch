import torch
from math import sqrt
from .true_positive_rate import TPR
from .false_positive_rate import FPR
from ._BaseMetric import _BaseMetric


class PT(_BaseMetric):

    def metric_func(self, logits: torch.Tensor, y: torch.Tensor, default: float = 0):
        """Compute the prevalence threshold

        Args:
            logits (torch.Tensor): output from the model (B, C)
            y (torch.Tensor): true labels (B, 1)
            default (float): Value to return if the metric is not well defined

        Returns:
            float: prevalence threshold of that batch
        """

        tpr = TPR()(logits, y)
        fpr = FPR()(logits, y)

        return (sqrt(tpr * fpr) - fpr) / (tpr - fpr) if (tpr - fpr) != 0 else default
