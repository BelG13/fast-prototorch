import torch
import torch.nn as nn

class ProtoTest(nn.Module):
    
    def __init__(self, in_features: int, out_features: int, n_classes: int):
        super().__init__()
        
        self.fc1 = nn.Linear(in_features=in_features, out_features=out_features)
        self.fc2 = nn.Linear(in_features=out_features, out_features=n_classes)
    
    
    def forward(self, input):
        h = nn.ReLu()(self.fc1(input))
        out = nn.ReLU()(self.fc2(h))
        
        return out
    
        