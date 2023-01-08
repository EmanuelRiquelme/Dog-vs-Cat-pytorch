import torch
import torch.nn as nn
class LeNet(nn.Module):
    def __init__(self,num_clases = 3):
        super(LeNet, self).__init__()
        self.layer1 = torch.nn.Conv2d(in_channels = 3, out_channels = 6,kernel_size = 5 , stride=1, padding=2)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(kernel_size = (2,2),stride = 2)
        self.layer2 = torch.nn.Conv2d(in_channels = 6,out_channels = 16, kernel_size = (5,5))
        self.fc1 = nn.Linear(400,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,num_clases)
    def forward(self,img):
        img = self.layer1(img)
        img = self.relu(img)
        img = self.maxpool(img)
        img = self.layer2(img)
        img = self.relu(img)
        img = self.maxpool(img).reshape(-1,400)
        img = self.fc1(img)
        img = self.relu(img)
        img = self.fc2(img)
        img = self.relu(img)
        img = self.fc3(img)
        return img
