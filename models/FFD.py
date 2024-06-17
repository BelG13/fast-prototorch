import torch.nn as nn

class FFD(nn.Module):
    
    def __init__(self, in_features: int, n_classes: int, name: str=None, flatten=False):
        super(FFD, self).__init__()
        
        self.name = name or 'ffd'   
        
        self.fc1 = nn.Linear(in_features=in_features, out_features=16)
        self.bn1 = nn.BatchNorm1d(num_features=16)
        
        self.fc2 = nn.Linear(in_features=16, out_features=32)
        self.bn2 = nn.BatchNorm1d(num_features=32)
        
        self.fc3 = nn.Linear(in_features=32, out_features=64)
        self.bn3 = nn.BatchNorm1d(num_features=64)
        
        self.fc4 = nn.Linear(in_features=64, out_features=32)
        self.bn4 = nn.BatchNorm1d(num_features=32)
        
        self.fc5 = nn.Linear(in_features=32, out_features=n_classes)
    
    
    def forward(self, x):
        h1 = nn.ReLU()(self.fc1(x))
        h1 = self.bn1(h1)
        
        h2 = nn.ReLU()(self.fc2(h1))
        h2 = self.bn2(h2)
        
        h3 = nn.ReLU()(self.fc3(h2))
        h3 = self.bn3(h3)
        
        h4 = nn.ReLU()(self.fc4(h3))
        h4 = self.bn4(h4)
        
        out = nn.Softmax(dim=1)(self.fc5(h4))
        # out = nn.Sigmoid()(self.fc5(h4))
        return out