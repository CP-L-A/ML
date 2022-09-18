#K-均值聚类基于numpy的实现
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
#加载数据集，采用sklearn自带的iris数据集
def load_data():
    data,label=load_iris(return_X_y=True)
    train_data,test_data,train_label,test_label=train_test_split(data,label,test_size=0.4,random_state=42)
    return train_data,train_label
#计算欧式距离
def calcEculd(vecA,vecB):
    return np.sqrt(np.sum(np.power(vecA-vecB,2)))
#构建初始的簇的质心，随机生成
#每个质心需要保证在数据内部，即质心的每个坐标应该在对应的特征的最大值和最小值之间
def randCent(dataSet,k):
    n=np.shape(dataSet)[1]                  #特征的数量
    centroids=np.mat(np.zeros((k,n)))       #构造质心的矩阵
    for j in range(n):
        minJ=np.min(dataSet[:,j])
        maxJ=np.max(dataSet[:,j])
        rangJ=float(maxJ)-float(minJ)
        centroids[:,j]=minJ+rangJ*np.random.rand(k,1)
    return centroids
#K均值训练过程
def K_means(dataSet,k):
    m=np.shape(dataSet)[0]
    clusterAssment=np.mat(np.zeros((m,2)))                #构造矩阵用于存放分类结果以及误差
    centroids=randCent(dataSet,k)                         #生成初始的随机质心
    clusterChange=True                                    #标志位，用于判别聚类结果是否发生了改变
    item=0
    while clusterChange:
        clusterChange=False
        for i in range(m):                                #对于数据集中的每个样本，计算相距最短的质心
            minDist=float('inf')
            minIndex=-1
            for j in range(k):
                disJi=calcEculd(centroids[j,:],dataSet[i,:])
                if disJi<minDist:
                    minDist=disJi
                    minIndex=j
            if clusterAssment[i,0]!=minIndex:               #如果分类结果与之前存放的信息不同则代表聚类结果发生了改变
                clusterChange=True
            clusterAssment[i,:]=minIndex,minDist**2         #存放新的聚类信息
        print(centroids)
        for cent in range(k):                               #更新质点信息
            ptsInclust=dataSet[np.nonzero(clusterAssment[:,0].A==cent)[0]]
            centroids[cent,:]=np.mean(ptsInclust,axis=0)
        item+=1
    return centroids,clusterAssment,item
if __name__=='__main__':
    data,label=load_data()
    orids,Assment,item=K_means(data,3)
    print(item)

