import torch
from .utils import _get_conditions
from .true_negative_rate import TNR
from .true_positive_rate import TPR
from ._BaseMetric import _BaseMetric


class BM(_BaseMetric):

    def metric_func(self, logits: torch.Tensor, y: torch.Tensor, default: float = None):
        """Compute the informedness.

        Args:
            logits (torch.Tensor): output from the model (B, C)
            y (torch.Tensor): true labels (B, 1)
            default (float): Value to return if the metric is not well defined

        Returns:
            float: informedness of that batch
        """

        tpr = TPR()(logits, y)
        tnr = TNR()(logits, y)

        return tpr + tnr - 1
