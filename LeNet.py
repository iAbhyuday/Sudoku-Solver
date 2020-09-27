import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LeNet(nn.Module):

    def __init__(self):
        super(LeNet,self).__init__()
        # Conv Block
        self.conv1 = nn.Conv2d(1,6,5,stride=1,padding=2,bias=True)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(6,16,5,stride=1,padding=0,bias=True)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # FC Block
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)


    def forward(self,x):

        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
