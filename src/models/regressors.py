import torch.nn as nn, torch.nn.functional as F

class SmallMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=1024, hidden2=1024, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden2)
        self.fc3 = nn.Linear(hidden2, out_dim)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        x = self.drop(F.gelu(self.fc1(x)))
        x = self.drop(F.gelu(self.fc2(x)))
        return self.fc3(x)