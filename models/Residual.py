import torch
import torch.nn as nn


class Residual(nn.Module):
    
    """fdd with residual connections
    """
    
    def __init__(self, in_features, transit_size, n_classes):
        super(Residual, self).__init__()
        
        self.fc1 = nn.Linear(in_features, transit_size)
        self.fc2 = nn.Linear(transit_size, transit_size)
        self.fc3 = nn.Linear(transit_size, transit_size)
        self.fc4 = nn.Linear(transit_size, transit_size)
        self.fc5 = nn.Linear(transit_size, transit_size)
        self.fc6 = nn.Linear(transit_size, transit_size)
        self.fc7 = nn.Linear(transit_size, transit_size)
        self.fc8 = nn.Linear(transit_size, transit_size)
        self.fc9 = nn.Linear(transit_size, transit_size)
        self.out = nn.Linear(transit_size, n_classes)
        
    
    def forward(self, input):
        
        h1 = nn.ReLU()(self.fc1(input))
        h2 = nn.ReLU()(self.fc2(h1))
        h3 = nn.ReLU()(self.fc3(h2)) + h1
        h4 = nn.ReLU()(self.fc4(h3))
        h5 = nn.ReLU()(self.fc5(h4)) + h3
        h6 = nn.ReLU()(self.fc6(h5))
        h7 = nn.ReLU()(self.fc7(h6)) + h5
        h8 = nn.ReLU()(self.fc8(h7))
        h9 = nn.ReLU()(self.fc9(h8)) + h7
        
        out = nn.Softmax(dim=1)(self.out(h9))
        
        return out
        