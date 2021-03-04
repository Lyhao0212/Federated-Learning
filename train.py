
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import random

def train_init(train_loader,arg):
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 20, 5, 1)
            self.conv2 = nn.Conv2d(20, 50, 5, 1)
            self.fc1 = nn.Linear(4 * 4 * 50, 500)
            self.fc2 = nn.Linear(500, 10)
            nn.init.normal_(self.conv1.weight)
            nn.init.normal_(self.conv2.weight)
            nn.init.normal(self.conv1.bias, val=0)
            nn.init.normal(self.conv2.bias, val=0)
            nn.init.normal_(self.fc1.weight)
            nn.init.normal_(self.fc2.weight)
            nn.init.normal(self.fc1.bias, val=0)
            nn.init.normal(self.fc2.bias, val=0)

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

                nn.init.normal_(self.layer.weight)
                nn.init.constant_(self.layer.bias, val=0)
            def forward(self, x):
                logit =self.layer(x)
                return logit

    epoches = arg.epochs
    lr = arg.lr
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if arg.dataset=='MNIST':
        input_size=28*28*1
        subset_size=600
    elif arg.dataset=="CIFAR10":
        input_size=32*32*3
        subset_size=500
    if arg.model=='LR':
        model=Logistic(input_size,10).train()
    elif arg.model=='CNN':
        model=CNN().train()
    model.to(device)
    loss_func = nn.CrossEntropyLoss()
    if arg.optimizer=="SGD":
        optimizer=optim.SGD(model.parameters(),lr=lr)
    elif arg.optimizer=="Nesterov":
        optimizer = optim.SGD(model.parameters(), lr=lr,momentum=0.95,nesterov=True)
    elif arg.optimizer=="Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    trainloss=0
    size=0
    for epoch in range(epoches):
        if arg.batch_size == 1:
            a = random.randint(0, subset_size-1)

            inputlist = []
            for i in train_loader:
                inputlist.append(i)
            input_sample = inputlist[a]
            images = input_sample[0]
            labels = input_sample[1]
            if arg.model == 'LR':
                images = images.reshape(-1, input_size).to(device)
            elif arg.model == 'CNN':
                images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            loss = loss_func(output, labels)
            trainloss+=loss.item()
            size+=1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            for images, labels in train_loader:

                if arg.model == 'LR':
                    images = images.reshape(-1, input_size).to(device)
                elif arg.model == 'CNN':
                    images = images.to(device)
                labels = labels.to(device)
                output = model(images)

                loss = loss_func(output, labels)

                #print(loss.item())
                trainloss += loss.item()
                size+=1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    #print('Training Loss: {:.4f}'.format( trainloss/epoches))
    params = list(model.named_parameters())

    return params,trainloss/size

def train(train_loader,arg,comparalist):
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
            self.layer.bias=Parameter(comparalist[1])
            #nn.init.constant_(self.layer.bias, val=0)
        def forward(self, x):
            logit = self.layer(x)
            return logit
    epoches = arg.epochs
    lr = arg.lr
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if arg.dataset=='MNIST':
        input_size=28*28*1
        subset_size=600
    elif arg.dataset=="CIFAR10":
        input_size=32*32*3
        subset_size=500
    if arg.model=='LR':
        model=Logistic(input_size,10).train()
    elif arg.model=='CNN':
        model=CNN().train()
    model.to(device)
    loss_func = nn.CrossEntropyLoss()
    if arg.optimizer=="SGD":
        optimizer=optim.SGD(model.parameters(),lr=lr)
    elif arg.optimizer=="Nesterov":
        optimizer = optim.SGD(model.parameters(), lr=lr,momentum=0.95,nesterov=True)
    elif arg.optimizer=="Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)

    trainloss=0
    size=0
    for epoch in range(epoches):

        if arg.batch_size==1:
            a = random.randint(0, subset_size-1)

            inputlist = []
            for i in train_loader:
                inputlist.append(i)
            input_sample = inputlist[a]
            images = input_sample[0]
            labels = input_sample[1]
            if arg.model == 'LR':
                images = images.reshape(-1, input_size).to(device)
            elif arg.model == 'CNN':
                images = images.to(device)
            labels = labels.to(device)
            output = model(images)

            loss = loss_func(output, labels)
            trainloss += loss.item()
            size+=1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            for images, labels in train_loader:
                if arg.model == 'LR':
                    images = images.reshape(-1, input_size).to(device)
                elif arg.model == 'CNN':
                    images = images.to(device)
                labels = labels.to(device)
                output = model(images)
                size+=1
                loss = loss_func(output, labels)
                trainloss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        #print('Training Loss: {:.4f}'.format(trainloss / epoches))


    params = list(model.named_parameters())
    #print(params)

    return params,trainloss/size


