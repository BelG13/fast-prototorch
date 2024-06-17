import torch

def recall(logits: torch.Tensor, y: torch.Tensor):
    """Compute the recall

    Args:
        logits (torch.Tensor): output from the classifier
        y (torch.Tensor): ground thrie labels

    Returns:
        float: the recall
    """
    
    preds = logits.argmax(dim=1)
    
    # true positive
    tp = (preds[y==1] == y[y==1]).int().sum().item()
    
    # false negative
    fn = (y[preds[y==1]] == 0).int().sum().item()
    
    try:
        return tp / (tp + fn)
    except ZeroDivisionError:
        raise("the predicted class is the same across the batch the recall is impossible to compute")


class RecallMetric(object):
    
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(RecallMetric, cls).__new__(cls)
        return cls.instance
    
    def __call__(self, logits, y):
        return recall(logits, y)
            
    

