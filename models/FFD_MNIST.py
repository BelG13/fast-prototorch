import torch.nn as nn

class FFD_MNIST(nn.Module):
    
    def __init__(self, in_features: int, n_classes: int, name: str=None):
        super(FFD_MNIST, self).__init__()
        
        self.name = name or 'ffd'   
        
        self.ft1 = nn.Flatten()
        self.fc1 = nn.Linear(in_features=in_features, out_features=4098)
        self.bn1 = nn.BatchNorm1d(num_features=4098)
        
        self.fc2 = nn.Linear(in_features=4098, out_features=4098)
        self.bn2 = nn.BatchNorm1d(num_features=4098)
        
        self.fc3 = nn.Linear(in_features=4098, out_features=4098)
        self.bn3 = nn.BatchNorm1d(num_features=4098)
        
        self.fc4 = nn.Linear(in_features=4098, out_features=2044)
        self.bn4 = nn.BatchNorm1d(num_features=2044)
        
        self.fc5 = nn.Linear(in_features=2044, out_features=n_classes)
    
    
    def forward(self, x):
        x  = self.ft1(x)
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