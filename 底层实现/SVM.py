#SVM实现代码
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.datasets
training_set=[[1,2],[2,3],[3,1],[4,2]]
label=[1,1,-1,-1]
dataset=np.mat(training_set)
labelmat=np.mat(label).T
class opstruct():
    def __init__(self,data,label,C,toler):
        self.X=data
        self.label=label
        self.C=C
        self.tol=toler
        self.m=np.shape(data)[0]
        self.alphas=np.mat(np.zeros((self.m,1)))
        self.eCache=np.mat(np.zeros((self.m,2)))
        self.b=0
        
def calcEk(OS,k):
    fxk=float(np.multiply(OS.alphas,OS.label).T*(OS.X*OS.X[k,:].T))+OS.b
    Ek=fxk-float(OS.label[k])
    return Ek

def selectJ(i,OS,Ei):
    maxK=-1
    maxDelatE=0
    Ej=0
    OS.eCache[i]=[1,Ei]
    validEcache=np.nonzero(OS.eCache[:,0].A)[0]
    if len(validEcache)>1: #如果存在两个以上的有效误差，则说明并非第一次循环，需要找到DelataE最大的点
        for k in validEcache:
           if k==i:continue 
           Ek=calcEk(OS, k)
           deltaE=np.abs(Ei-Ek)
           if deltaE>maxDelatE:
               maxDelatE=deltaE
               maxK=k
               Ej=Ek
        return maxK,Ej
    else:                #如果只有一个有效误差，则为第一次循环，所有的alpha都一样，随机选择另一个alpha进行优化
        maxK=i
        while (i==maxK):
            maxK=int(random.uniform(0,OS.m))
        Ej=calcEk(OS,maxK)
        return maxK,Ej
    
def updateEk(OS,k):
    Ek=calcEk(OS,k)
    OS.eCache[k]=[1,Ek]


def clipAlpha(alphaJ,L,H):     #对a2进行剪辑
    if alphaJ<=H and alphaJ>=L:
        return alphaJ
    elif alphaJ>H:
        return H
    else:
        return L
    
def SMO(i,OS):                                   #SMO优化算法,每次优化若更改了alpha则返回1，不更改则返回0
    Ei=calcEk(OS,i)
    #判断是否违反KKT条件
    #在误差toler的范围内，若a1>0,则KKT条件为yi*Ei<tol,若a1<C,则kkt条件为yi*Ei>-tol
    if ((OS.label[i]*Ei<-OS.tol) and (OS.alphas[i]<OS.C)) or ((OS.label[i]*Ei>OS.tol) and (OS.alphas[i]>0)):
        j,Ej=selectJ(i, OS, Ei)                 #选择第二个优化点
        alphaIold=OS.alphas[i].copy()
        alphaJold=OS.alphas[j].copy()
        #y1和y2是否相等确定L和H
        if (OS.label[i]!=OS.label[j]):
            L=max(0,OS.alphas[j]-OS.alphas[i])
            H=min(OS.C,OS.C+OS.alphas[j]-OS.alphas[i])
        else:
            L=max(0,OS.alphas[j]+OS.alphas[i]-OS.C)
            H=min(OS.C,OS.alphas[j]+OS.alphas[i])
        if L==H:
            return 0
        #计算x1与x2两点的距离
        eta=OS.X[i,:]*OS.X[i,:].T+OS.X[j,:]*OS.X[j,:].T-2*OS.X[i,:]*OS.X[j,:].T
        if eta<=0:
            return 0
        #计算新的a2的值，并根据L和H进行剪辑
        OS.alphas[j]+=OS.label[j]*(Ei-Ej)/eta
        OS.alphas[j]=clipAlpha(OS.alphas[j], L, H)
        updateEk(OS, j)                       #计算结束之后更新对应的误差
        if (abs(OS.alphas[j]-alphaJold)<0.0001):     #如果前后变化幅度过小则返回
            return 0
        OS.alphas[i]+=OS.label[j]*OS.label[i]*(alphaJold-OS.alphas[j])    #计算a1
        updateEk(OS, i)    #更新a1的误差
        #计算b1和b2
        b1=OS.b-Ei-OS.label[i]*(OS.X[i,:]*OS.X[i,:].T)*(OS.alphas[i]-alphaIold)-\
           OS.label[j]*(OS.X[j,:]*OS.X[i,:].T)*(OS.alphas[j]-alphaJold)
        b2=OS.b-Ej-OS.label[j]*(OS.X[j,:]*OS.X[j,:].T)*(OS.alphas[j]-alphaJold)-\
           OS.label[i]*(OS.X[i,:]*OS.X[j,:].T)*(OS.alphas[i]-alphaIold)
        if (OS.alphas[i]<OS.C) and (OS.alphas[i]>0):           #选择不违反kkt条件的b作为新的b值
            OS.b=b1
        elif (OS.alphas[j]<OS.C) and (OS.alphas[j]>0):
            OS.b=b2
        else:
            OS.b=(b1+b2)/2
        return 1
    else:
        return 0
    
def SVM_training(data,label,maxiter,toler,C,Ktup=('lin',0)):    #SVM算法框架，输入数据，标签，最大迭代次数，误差以及C
    OS=opstruct(data,label,C,toler)
    it=0
    entireSet=True                                           #标志,确定是否遍历全部数据
    alphaChange=0                                            #标志，用于判别alpha是否发生了改变
    while (it<maxiter) and (alphaChange>0 or entireSet):     #当迭代次数小于最大迭代次数且alpha进行过修改或已经遍历过所有非边界值时
        alphaChange=0                                        #每次迭代前将alphaChange置0
        if entireSet:                                        #遍历所有数据点，对于初次迭代，所有a=0，应
            for i in range(OS.m):
                alphaChange+=SMO(i, OS)
            it+=1
        else:                                                #遍历间隔上的边界点
            bounds=np.nonzero((OS.alphas.A>0)*(OS.alphas.A<OS.C))[0]
            for i in bounds:
                alphaChange+=SMO(i, OS)
            it+=1
        if entireSet:                                       #如果遍历过所有的数据，则下次遍历间隔边界上的点
            entireSet=False
        elif (alphaChange==0):                              #如果间隔边界上的点均满足KKT条件则遍历所有数据
            entireSet=True
        
    w=calcW(data, OS.alphas, label)
    return w,OS.b

def calcW(data,alpha,labels):
    m,n=np.shape(data)
    w=np.zeros((n,1))
    for i in range(m):
        w+=np.multiply(alpha[i]*labelmat[i],data[i,:].T)
    return w


    
if __name__=='__main__':
    start=time.perf_counter()
    w,b=SVM_training(dataset, labelmat, 1000, 0.001, 0.4)
    end=time.perf_counter()
    print('运行时间'+str(end-start))
    w1=round(float(w[0][0]),2)
    w2=round(float(w[1][0]),2)
    b_=round(float(b),2)
    print('训练得到的模型为：')
    print(str(w1)+'x1'+'+'+str(w2)+'x2'+'+'+str(float(b_)))
    print('\n')
    figure=plt.figure()
    ax=figure.add_subplot(111)
    n=np.shape(dataset)[0]
    x=dataset[:,0].A
    y=dataset[:,1].A
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    for i in range(n):
        if labelmat[i]==1:
            plt.scatter(x[i],y[i],color='red')
        else:
            plt.scatter(x[i],y[i],color='blue')
    w1=float(w[0][0])
    w2=float(w[1][0])
    x2=np.linspace(-5, 5)
    y2=np.multiply((-w1/w2),x2)
    for item in y2:
        item+=b
    plt.plot(x2,y2)
    plt.show()
    
    
