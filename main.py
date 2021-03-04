import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data.dataloader as dataloader
from torch.utils.data import sampler,Subset
import numpy as np
import random
from random import sample
import matplotlib.pyplot as plt
import time
from matplotlib.pyplot import MultipleLocator
from config import Arguments
from train import train,train_init
from test import test
from Q import Qfun,Qfun_paralist
from com_para import combine_params_init,combine_params

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    set_seed(2021)
    # 设置并打印参数
    arg = Arguments()
    max_length = max([len(key) for key in arg.__dict__.keys()])
    fmt_string = '\t%' + str(max_length) + 's : %s'
    print('>>> Arguments:')
    for keyPair in sorted(arg.__dict__.items()):
        print(fmt_string % keyPair)

    # 数据准备
    Datalist = []
    train_set = torchvision.datasets.MNIST(root="./data", train=True, transform=transforms.ToTensor(), download=True)
    random_sampler = sampler.RandomSampler(data_source=train_set) #打乱数据的索引
    batch_sampler = sampler.BatchSampler(random_sampler, int(len(train_set) / arg.K), True) #生成一定数量的索引的集合
    for i in batch_sampler:
        Datalist.append(dataloader.DataLoader(Subset(train_set, i), batch_size=arg.batch_size, shuffle=True)) #按照索引生成子数据集
    test_set = torchvision.datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor(), download=True)
    test_loader = dataloader.DataLoader(dataset=test_set,batch_size=arg.test_batch_size ,shuffle=True)
    clients_num = int(arg.K * arg.C)
    datalist = sample(Datalist, clients_num) #从子数据集中抽取若干个

    # 开始训练
    print(">>> Training...")
    print("=========================================================================================================")
    acclist = []
    train_losslist = []
    test_losslist = []
    starttime = time.time()

    # 初始训练并量化
    train_loss = 0
    paralist = []
    for i in range(clients_num):
        para, trainloss = train_init(datalist[i], arg)
        paralist.append(Qfun(para, arg.g,arg.B))
        train_loss += trainloss / clients_num
    # 上传参数并融合
    comparalist = combine_params_init(paralist)
    # 测试
    acc, loss = test(test_loader, comparalist, arg)

    print(">>> Round:   {} / Acc:   {}% /Train_loss:   {} / Test_loss:   {} / Time:   {}s / Learning_rate:   {}"
            .format(1,acc,round(train_loss,4),round(loss,4),round(time.time() - starttime,3),arg.lr))
    print("=========================================================================================================")
    acclist.append(acc)
    test_losslist.append(loss)
    train_losslist.append(train_loss)
    for i in range(arg.Round - 1):
        starttime = time.time()
        datalist = sample(Datalist, int(arg.C * arg.K))
        old_com_paralist = []
        for paras in comparalist:
            old_com_paralist.append(paras.clone())
        # 参数量化
        comparalist = Qfun_paralist(comparalist,arg.g, arg.B)
        # 下载参数、训练并量化

        print("量化后精度", test(test_loader, comparalist, arg))
        train_loss = 0
        paralist=[]
        for j in range(clients_num):
            com_paralist=[]
            for paras in comparalist:
                com_paralist.append(paras.clone())
            para, trainloss = train(datalist[j], arg, com_paralist)
            paralist.append(Qfun(para, arg.g,arg.B))
            train_loss += trainloss / clients_num

        # 上传参数并融合
        comparalist = combine_params(paralist, arg.gamma, old_com_paralist)

        # 测试
        acc, loss = test(test_loader, comparalist, arg)


        print(
            ">>> Round:   {} / Acc:   {}% / Train_loss:   {} / Test_loss:   {} / Time:   {}s / Learning_rate:   {}".format(
                i + 2, acc, round(train_loss, 4), round(loss, 4), round(time.time() - starttime, 3), arg.lr))
        print(
            "=========================================================================================================")






        acclist.append(acc)
        test_losslist.append(loss)
        train_losslist.append(train_loss)
        if (i + 2) % 20 == 0 and arg.lr >= 0.0001:
            arg.lr = arg.lr * arg.decay
        if (i + 2) % 100 == 0:
            print(test_losslist)
            print(acclist)
            print(train_losslist)
    print("Max acc = ",max(acclist))
    print(acclist.index(max(acclist)))







    # 设置画布，创建子图
    p1 = plt.figure(figsize=(8, 6), dpi=80)
    x = np.arange(0, len(test_losslist)).astype(dtype=np.str)
    ax1 = p1.add_subplot(3, 1, 1)
    plt.title("SQ/B= {},g= {},γ= {}".format(arg.B,arg.g,arg.gamma))
    plt.ylabel("Test_loss")
    plt.xlabel("Number of combination")
    x_major_locator = MultipleLocator(50)
    y_major_locator = MultipleLocator(0.2)
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    plt.plot(x, test_losslist)

    # 第2个子图ax2
    ax2 = p1.add_subplot(3, 1, 2)
    x = np.arange(0, len(acclist)).astype(dtype=np.str)
    plt.ylabel("Test_acc")
    plt.xlabel("Number of combination")

    x_major_locator = MultipleLocator(50)
    y_major_locator = MultipleLocator(10)
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    plt.plot(x, acclist)
    # 第3个子图ax2
    ax3 = p1.add_subplot(3, 1, 3)
    x = np.arange(0, len(train_losslist)).astype(dtype=np.str)
    plt.ylabel("Train_loss")
    plt.xlabel("Number of combination")

    x_major_locator = MultipleLocator(50)
    y_major_locator = MultipleLocator(10)
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    plt.plot(x, train_losslist)

    plt.show()
if __name__=="__main__":
    main()