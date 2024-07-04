import torch
from abc import ABC, abstractmethod

class _BaseMetric(ABC):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(_BaseMetric, cls).__new__(cls)
        return cls.instance

    @abstractmethod
    def metric_func(self, logits: torch.Tensor, y: torch.Tensor, default: float):
        pass
    
    def __call__(self, *args, **kwargs):
        return self.metric_func(*args, **kwargs)