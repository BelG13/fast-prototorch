import torch
from .true_negative_rate import TNR
from .false_negative_rate import FNR
from ._BaseMetric import _BaseMetric


class LRMinus(_BaseMetric):

    def metric_func(self, logits: torch.Tensor, y: torch.Tensor, default: float = 0):
        """Compute the negative likelihood ratio

        Args:
            logits (torch.Tensor): output from the model (B, C)
            y (torch.Tensor): true labels (B, 1)
            default (float): Value to return if the metric is not well defined

        Returns:
            float: negative likelihood ratio of that batch
        """

        tnr = TNR()(logits, y)
        fnr = FNR()(logits, y)

        return tnr / fnr if fnr != 0 else default
