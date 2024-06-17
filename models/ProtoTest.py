import torch
import torch.nn as nn

class ProtoTest(nn.Module):
    
    def __init__(self, in_features: int, out_features: int, n_classes: int, name: str=None):
        super(ProtoTest, self).__init__()
        
        self.name = name or 'Prototype'
        self.fc1 = nn.Linear(in_features=in_features, out_features=out_features)
        self.fc2 = nn.Linear(in_features=out_features, out_features=n_classes)
    
    
    def forward(self, x):
        h = nn.ReLU()(self.fc1(x))
        out = nn.Softmax(dim=1)(self.fc2(h))
        return out
    
        