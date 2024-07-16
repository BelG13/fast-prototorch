import torch

def _get_conditions(preds: torch.Tensor, y: torch.tensor):
    """Compute the condition of each output.

    Args:
        preds (torch.Tensor): predicted labels (B, 1)
        y (torch.tensor): ground truth label (B, 1)

    Returns:
        tuple: tp, fp, tn, fn
    """
    return (
        ((preds == 1) & (y == 1)).int().sum().item(), # hit
        ((preds == 0) & (y == 1)).int().sum().item(), # false alarm
        ((preds == 0) & (y == 0)).int().sum().item(), # correct rejection
        ((preds == 1) & (y == 0)).int().sum().item(), # miss
    )