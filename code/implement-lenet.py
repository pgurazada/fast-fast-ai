import torch 
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, 5) # since images are black and white there is only one channel
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == '__main__':
    net = Net()

    print(net)

    #params = list(net.parameters())
    #print(len(params))
    #print(params[0].size())

    input_ = torch.randn(1, 1, 32, 32)
    target = torch.randn(1, 10)
    target = target.view(1, -1)
    print(input_.shape)
    
    out = net(input_)
    print(out)

    net.zero_grad()
    out.backward(torch.randn(1, 10))

    criterion = nn.MSELoss()
    loss = criterion(out, target)
    print(loss)


