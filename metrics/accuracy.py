import torch
from ._BaseMetric import _BaseMetric


class Accuracy(_BaseMetric):

    def metric_func(self, logits: torch.Tensor, y: torch.Tensor, default: float = None):
        """Compute the accuracy of a given batch

        Args:
            logits (torch.Tensor): output from the model (B, C)
            y (torch.Tensor): true labels (B, 1)
            default (float): Value to return if the metric is not well defined

        Returns:
            float: accuracy score
        """
        preds = logits.argmax(dim=1)
        return (preds == y).int().sum(dim=0).item() / len(preds)
