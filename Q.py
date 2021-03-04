import torch
import numpy as np

def SQ_old(x, B):
    if x > 0:
        a = 0.5 * x
        b = 2 * x
        delta = (b - a) / (B - 1)
        list1 = []
        t1 = a
        while (t1 <= b):
            list1.append(t1)
            t1 += delta
        if max(list1) < b:
            list1.append(b)
        print(list1)
        for i in range(len(list1)):
            if list1[i] <= x and list1[i + 1] >= x:
                m = list1[i]
                n = list1[i + 1]
        p = (x - m) / delta
        if np.random.binomial(1, p, 1):
            return n

        else:
            return m
    elif x==0:
        return x
    else:
        a = 0.5 * x
        b = 2 * x
        delta = (a-b) / (B - 1)
        list1 = []
        t1 = b

        while (t1 <= a):
            list1.append(t1)
            t1 += delta
        if max(list1) < a:
            list1.append(a)
        for i in range(len(list1)):
            if list1[i] <= x and list1[i + 1] >= x:
                m = list1[i]
                n = list1[i + 1]
        p = (x - m) / delta
        if np.random.binomial(1, p, 1):
            return n

        else:
            return m

def NQ(x,g):
    G=2**(g-1)
    x=x*G
    x=round(x)
    if x>G-1:
        x=G-1
    elif x<-G:
        x=-G
    else:
        x=x
    return x/G

def SQ(x,g,B):
    G=2**(g-1)
    x=x*G
    x=round(x)
    if x>G-1:
        x=G-1
        #print("大")
        return x/G
    elif x<-G:
        x=-G
        #print("小")
        return x/G
    else:
        a = -G
        b = G-1
        delta = (b - a) / (B - 1)
        list1 = []
        t1 = a
        while(t1<=b):
            list1.append(t1)
            t1+=delta
        if max(list1)<b:
            list1.append(b)

        #print(list1)
        for i in range(len(list1)):
            if list1[i] <= x and list1[i + 1] >= x:
                m = list1[i]
                n = list1[i + 1]
                break
        p = (x - m) / delta
        if np.random.binomial(1, p, 1):
            return n/G

        else:
            return m/G


def Qfun(para,g,B):
    for i in range(0,len(para)):
        temppara=para[i][1].data
        shape=temppara.shape
        temppara=temppara.reshape(-1,1)
        Qpara=[]
        for j in temppara:
            a=j.cpu().numpy()
            b=SQ(float(a),g,B)
            Qpara.append(b)
        Qpara=np.array(Qpara)
        Qpara=torch.tensor(Qpara)
        Qpara=Qpara.type(torch.FloatTensor)
        Qpara=Qpara.cuda()
        Qpara=Qpara.reshape(shape)
        para[i][1].data=Qpara
    return para

def Qfun_paralist(para,g,B):
    for i in range(len(para)):
        temppara=para[i]
        shape=temppara.shape
        temppara=temppara.reshape(-1,1)
        Qpara=[]
        for j in temppara:
            a=j.cpu().numpy()
            b=SQ(float(a),g,B)
            Qpara.append(b)
        Qpara=np.array(Qpara)
        Qpara=torch.tensor(Qpara)
        Qpara=Qpara.type(torch.FloatTensor)
        Qpara=Qpara.cuda()
        Qpara=Qpara.reshape(shape)
        para[i]=Qpara
    return para

def Qfun_gradlist(para,g,B):
    for i in range(len(para)):
        temppara=para[i]
        shape=temppara.shape
        temppara=temppara.reshape(-1,1)
        Qpara=[]
        for j in temppara:
            a=j.cpu().detach().numpy()
            b=SQ(float(a),g,B)
            Qpara.append(b)
        Qpara=np.array(Qpara)
        Qpara=torch.tensor(Qpara)
        Qpara=Qpara.type(torch.FloatTensor)
        Qpara=Qpara.cuda()
        Qpara=Qpara.reshape(shape)
        para[i]=Qpara
    return para


def NQfun(para,g):
    for i in range(0,len(para),2):
        temppara=para[i][1].data
        shape=temppara.shape
        temppara=temppara.reshape(-1,1)
        Qpara=[]
        for j in temppara:
            a=j.cpu().numpy()
            b=NQ(float(a),g)
            Qpara.append(b)
        Qpara=np.array(Qpara)
        Qpara=torch.tensor(Qpara)
        Qpara=Qpara.type(torch.FloatTensor)
        Qpara=Qpara.cuda()
        Qpara=Qpara.reshape(shape)
        para[i][1].data=Qpara
    return para

def NQfun_paralist(para,g):
    for i in range(len(para)):
        temppara=para[i]
        shape=temppara.shape
        temppara=temppara.reshape(-1,1)
        Qpara=[]
        for j in temppara:
            a=j.cpu().numpy()
            b=NQ(float(a),g)
            Qpara.append(b)
        Qpara=np.array(Qpara)
        Qpara=torch.tensor(Qpara)
        Qpara=Qpara.type(torch.FloatTensor)
        Qpara=Qpara.cuda()
        Qpara=Qpara.reshape(shape)
        para[i]=Qpara
    return para
if __name__=="__main__":

    print(SQ_old(0.99,11))
    print(SQ(-0.54266,10,10))