import torch
from .utils import _get_conditions
from ._BaseMetric import _BaseMetric


class FPR(_BaseMetric):

    def metric_func(self, logits: torch.Tensor, y: torch.Tensor, default: float = 0):
        """Compute the false positive rate

        Args:
            logits (torch.Tensor): output from the model (B, C)
            y (torch.Tensor): true labels (B, 1)
            default (float): Value to return if the metric is not well defined

        Returns:
            float: false positive rate of that batch
        """

        # predicted values
        preds = logits.argmax(dim=1)

        # conditions
        tp, fp, tn, fn = _get_conditions(preds, y)

        return fp / (tn + fp) if (tn + fp) > 0 else default
