#逻辑斯蒂回归实现
#采用梯度上升算法进行优化
import numpy as np
import matplotlib.pyplot as plt
def load_data():
    dataMat=[[1,2],[2,3],[3,1],[4,2]]
    for i in range(len(dataMat)):
        dataMat[i].append(1)
    labels=[1,1,0,0]
    return np.mat(dataMat),np.mat(labels).T
def sigmoid(inx):
    return 1/(1+np.exp(-inx))
def gradAscent(datamat,labelmat,maxIter,eta=0.01):
    m,n=np.shape(datamat)
    weight=np.mat(np.zeros((1,n)))
    for i in range(maxIter):
        grad=0
        for j in range(m):
            error=labelmat[j]-sigmoid(datamat[j]*weight.T)
            grad+=np.multiply(datamat[j],error)
        weight+=eta*grad
    return weight
if __name__=='__main__':
    x_train,y_train=load_data()
    w=gradAscent(x_train,y_train,500).T
    print(w)
    x_1=np.array(x_train[:,0])
    x_2=np.array(x_train[:,1])
    for i in range(len(x_1)):
        if float(y_train[i])==1:
            plt.scatter(x_1[i],x_2[i],color='red')
        else:
            plt.scatter(x_1[i], x_2[i], color='blue')
    y=-1*float(w[0])/float(w[1])*x_1+float(w[2])
    plt.plot(x_1,y)
    plt.show()

