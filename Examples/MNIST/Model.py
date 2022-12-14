import torch
import torch.nn as nn


class MNIST_2NN(nn.Module):
    def __init__(self):
        super(MNIST_2NN, self).__init__()
        self.linear1 = nn.Linear(784, 200)
        self.linear2 = nn.Linear(200, 200)
        self.linear3 = nn.Linear(200, 10)

    def forward(self, x):
        x1 = torch.relu(self.linear1(x))
        x2 = torch.relu(self.linear2(x1))
        y = self.linear3(x2)
        return y


class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        # [32, 24, 24]

        self.max_pool1 = nn.MaxPool2d(2)
        # [32, 12, 12]

        self.conv2 = nn.Conv2d(32, 64, 5)
        # [64, 8, 8]

        self.max_pool2 = nn.MaxPool2d(2)
        # [64, 4, 4]

        self.linear1 = nn.Linear(64 * 4 * 4, 512)
        self.linear2 = nn.Linear(512, 10)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.max_pool1(x1)
        x3 = self.conv2(x2)
        x4 = self.max_pool2(x3)
        x5 = torch.relu(self.linear1(torch.flatten(x4, 1)))
        y = self.linear2(x5)
        return y


# 2NN的升级版，因为我怕对于FedPer2NN太简单了，所以加了一层
# 但实际上2NN应用到FedPer上效果也不错
class MNIST_3NN(nn.Module):
    def __init__(self):
        super(MNIST_3NN, self).__init__()
        self.linear1 = nn.Linear(784, 200)
        self.linear2 = nn.Linear(200, 200)
        self.linear3 = nn.Linear(200, 200)
        self.linear4 = nn.Linear(200, 10)

    def forward(self, x):
        x1 = torch.relu(self.linear1(x))
        x2 = torch.relu(self.linear2(x1))
        x3 = torch.relu(self.linear3(x2))
        y = self.linear4(x3)
        return y
