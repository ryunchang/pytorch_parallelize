import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

from utils import *

columns = 6
rows = 6

# CPU 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(3, 8, 5)     #in, out, filtersize
        # self.pool = nn.MaxPool2d(2, 2)  #2x2 pooling
        # self.conv2 = nn.Conv2d(20, 24, 5)
        # self.fc1 = nn.Linear(60 * 53 * 53, 1000)
        # self.fc2 = nn.Linear(1000, 101)
        self.conv1 = nn.Conv2d(3, 2, 5)     #in, out, filtersize
        self.pool = nn.MaxPool2d(2, 2)  #2x2 pooling
        self.conv2 = nn.Conv2d(10, 6, 5)
        self.fc1 = nn.Linear(30 * 53 * 53, 1000)
        self.fc2 = nn.Linear(1000, 101)


    def forward(self, x):
        rcv = packet_receiver()
        x = _pickle.loads(rcv).to('cpu')
        x = self.conv1(x)
        packet_sender(x)
        rcv = packet_receiver()
        x = _pickle.loads(rcv).to('cpu')
        x = self.conv2(x)
        packet_sender(x)
        
        # rcv = packet_receiver()
        # x = pickle.loads(rcv).to('cpu')

        return x
