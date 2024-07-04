import torch
from .precision import Precision
from .negative_predictive_value import NPV
from ._BaseMetric import _BaseMetric


class MK(_BaseMetric):

    def metric_func(self, logits: torch.Tensor, y: torch.Tensor, default: float = None):
        """Compute the markedness

        Args:
            logits (torch.Tensor): output from the model (B, C)
            y (torch.Tensor): true labels (B, 1)
            default (float): Value to return if the metric is not well defined

        Returns:
            float: markedness of that batch
        """

        ppv = Precision()(logits, y)
        npv = NPV()(logits, y)

        return ppv + npv - 1
