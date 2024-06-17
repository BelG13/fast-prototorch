import torch
import torch.nn as nn
from torchsummary import summary


class ProtoCNN(nn.Module):
    
    def __init__(self, n_classes):
        
        super(ProtoCNN, self).__init__()
        
        self.c1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3)
        self.c2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.c3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
        self.c4 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3)
        
        self.fc1 = nn.Linear(in_features=288, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=2048)
        self.fc3 = nn.Linear(in_features=2048, out_features=n_classes)
        
        self.m1 = nn.MaxPool2d(kernel_size=2)
        self.m4 = nn.MaxPool2d(kernel_size=2)
        
        self.flat = nn.Flatten()
        
    def forward(self, x):
        
        h1 = self.c1(x)
        h1 = self.m1(h1)
        h1 = nn.ReLU()(h1)
        
        h2 = self.c2(h1)
        h2 = nn.ReLU()(h2)
        
        h3 = self.c3(h2)
        h3 = nn.ReLU()(h3)
        
        h4 = self.c4(h3)
        h4 = self.m4(h4)
        h4 = nn.ReLU()(h4)
        
        h4 = self.flat(h4)
        h5 = nn.ReLU()(self.fc1(h4))
        h6 = nn.ReLU()(self.fc2(h5))
        h7 = self.fc3(h6)
        
        return nn.Softmax(dim=1)(h7)
        
        
if __name__ == "__main__":
    summary(ProtoCNN(in_dim=28), (1, 28, 28))
        