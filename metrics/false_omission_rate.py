import torch
from .utils import _get_conditions
from ._BaseMetric import _BaseMetric


class FOR(_BaseMetric):

    def metric_func(self, logits: torch.Tensor, y: torch.Tensor, default: float = 0):
        """Compute the false omission rate (FOR)

        Args:
            logits (torch.Tensor): output from the model (B, C)
            y (torch.Tensor): true labels (B, 1)
            default (float): Value to return if the metric is not well defined

        Returns:
            float: The false omission rate of that batch
        """

        # predicted values
        preds = logits.argmax(dim=1)

        # conditions
        _, _, tn, fn = _get_conditions(preds, y)

        return fn / (tn + fn) if (tn + fn) > 0 else default
