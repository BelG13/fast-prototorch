import torch

def accuracy(logits: torch.Tensor, y: torch.Tensor):
    """_summary_

    Args:
        logits (torch.Tensor): output from the model
        y (torch.Tensor): true labels

    Returns:
        _type_: accuracy score
    """
    preds = logits.argmax(dim=1)
    return (preds == y).int().sum(dim=0).item() / len(preds)


class AccuracyMetric(object):
    
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(AccuracyMetric, cls).__new__(cls)
        return cls.instance
    
    def __call__(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Any:
        return accuracy(logits, y)