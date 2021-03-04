import torch

def combine_params_init(paralist):
    com_paralist=[]
    for j in range(0,len(paralist[0])):
        com_paralist.append(torch.zeros_like(paralist[0][j][1].data))
    for j in range(0,len(paralist[0])):
        for i in range(len(paralist)):
            com_paralist[j]+=paralist[i][j][1].data
    for i in range(len(com_paralist)):
        com_paralist[i]=com_paralist[i]/len(paralist)

    return com_paralist

def combine_params(paralist,gamma,old_com_paralist):

    com_paralist=[]
    for j in range(0,len(paralist[0])):
        com_paralist.append(torch.zeros_like(paralist[0][j][1].data))
    for j in range(0,len(paralist[0])):
        for i in range(len(paralist)):
            com_paralist[j]+=paralist[i][j][1].data

    for i in range(len(com_paralist)):
        com_paralist[i]=gamma*com_paralist[i]/len(paralist)+(1-gamma)*old_com_paralist[i]

    return com_paralist



def grad_calc(paralist,comparalist,lr):

    gradlist=[]
    for i in range(len(paralist)):
        for j in range(len(paralist[i])):
            gradlist.append((comparalist[j]-paralist[i][j][1])/lr)
    weight=torch.zeros_like(gradlist[0])
    bias=torch.zeros_like(gradlist[1])
    for i in range(0,len(gradlist),2):
        weight+=gradlist[i]
    weight=weight/(len(gradlist)/2)

    for i in range(1,len(gradlist),2):
        bias+=gradlist[i]
    bias=bias/(len(gradlist)/2)
    grad=[]
    grad.append(weight)
    grad.append(bias)
    return grad
