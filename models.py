import torch 
from torch import nn
import numpy as np

class teacher_model(nn.Module):
    def __init__(self):
        super(teacher_model, self).__init__()
        self.fc1 = nn.Linear(2, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x/5

class Net(nn.Module):

    def __init__(self):
        self.m = 512
        super(Net, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, self.m),
            nn.ReLU(),
            nn.Linear(self.m, 1),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x) / self.m
        return logits

class wideNet(nn.Module):

    def __init__(self):
        self.m = 1024*1024
        super(wideNet, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, self.m),
            nn.ReLU(),
            nn.Linear(self.m, 1),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x) / self.m
        return logits
