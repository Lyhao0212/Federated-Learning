class Arguments():
    def __init__(self):
        self.batch_size = 1
        self.test_batch_size = 10000
        self.epochs =5
        self.lr = 1
        self.decay=0.5
        self.Round=300
        self.C=0.1
        self.K=100
        self.B=15
        self.gamma=0.8
        self.g=10
        self.optimizer="SGD"  #SGD/Nesterov/Adam
        self.model="LR"   #LR/CNN
        self.dataset="MNIST"  #CIFAR10/MNIST
