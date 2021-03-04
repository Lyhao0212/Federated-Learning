import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

def test(test_loader,comparalist,arg):
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 20, 5, 1)
            self.conv2 = nn.Conv2d(20, 50, 5, 1)
            self.fc1 = nn.Linear(4 * 4 * 50, 500)
            self.fc2 = nn.Linear(500, 10)

            self.conv1.weight=Parameter(comparalist[0])
            self.conv2.weight=Parameter(comparalist[1])
            nn.init.constant_(self.conv1.bias, val=0)
            nn.init.constant_(self.conv2.bias, val=0)
            self.fc1.weight=Parameter(comparalist[2])
            self.fc2.weight=Parameter(comparalist[3])
            nn.init.constant_(self.fc1.bias, val=0)
            nn.init.constant_(self.fc2.bias, val=0)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2, 2)
            x = x.view(-1, 4 * 4 * 50)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x


    class Logistic(nn.Module):
        def __init__(self, in_dim, out_dim):
            super(Logistic, self).__init__()
            self.layer = nn.Linear(in_dim, out_dim)

            self.layer.weight=Parameter(comparalist[0])
            #nn.init.constant_(self.layer.bias, val=0)
            self.layer.bias=Parameter(comparalist[1])
        def forward(self, x):
            logit = self.layer(x)
            return logit
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if arg.dataset=='MNIST':
        input_size=28*28*1
    elif arg.dataset=="CIFAR10":
        input_size=32*32*3
    if arg.model == 'LR':
        model = Logistic(input_size, 10).eval()
    elif arg.model == 'CNN':
        model = CNN().eval()
    model.to(device)

    loss_func = nn.CrossEntropyLoss()
    # 测试，评估准确率
    correct = 0
    total = 0
    size=0
    loss=0
    for images, labels in test_loader:


        if arg.model == 'LR':
            images = images.reshape(-1, input_size).to(device)
        elif arg.model == 'CNN':
            images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        loss += loss_func(output, labels).item()
        values, predicte = torch.max(output, 1)
        total += labels.size(0)
        size+=1
        correct += (predicte == labels).sum().item()
    return 100 * correct / total,loss/size