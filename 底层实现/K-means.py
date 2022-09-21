#K-均值聚类基于numpy的实现
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
#加载数据集，采用sklearn自带的iris数据集
def load_data():
    data,label=load_iris(return_X_y=True)
    return data,label
#计算欧式距离
def calcEculd(vecA,vecB):
    return np.sqrt(np.sum(np.power(vecA-vecB,2)))
#构建初始的簇的质心，随机生成。随机生成的结果会影响算法的收敛性
#每个质心需要保证在数据内部，即质心的每个坐标应该在对应的特征的最大值和最小值之间
def randCent(dataSet,k):
    np.random.seed(42)
    n=np.shape(dataSet)[1]                  #特征的数量
    centroids=np.mat(np.zeros((k,n)))       #构造质心的矩阵
    for j in range(n):
        minJ=np.min(dataSet[:,j])
        maxJ=np.max(dataSet[:,j])
        rangJ=float(maxJ)-float(minJ)
        centroids[:,j]=minJ+rangJ*np.random.rand(k,1)
    return centroids
#K均值训练过程
#返回各个簇的质心坐标与划分结果矩阵
def K_means(dataSet,k):
    m=np.shape(dataSet)[0]
    clusterAssment=np.mat(np.zeros((m,2)))                #构造矩阵用于存放分类结果以及误差
    centroids=randCent(dataSet,k)                         #生成初始的随机质心
    clusterChange=True                                    #标志位，用于判别聚类结果是否发生了改变
    item=0                                                #显示迭代次数
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
        #更新质心信息，具体步骤如下
        #对于每个质心，提取出每个质心所包含的数据集
        #计算特征的平均值，作为新的质心
        for cent in range(k):
            ptsInclust=dataSet[np.nonzero(clusterAssment[:,0].A==cent)[0]]
            centroids[cent,:]=np.mean(ptsInclust,axis=0)   #对每一行求平均值
        item+=1
    return centroids,clusterAssment
#二分K均值算法
def Bin_K_means(datamat,k=2):
    global bewtNewCents, bestCentTosplit, bestClutsAss
    if k<=1:
        return
    m=np.shape(datamat)[0]
    clusterAssment=np.mat(np.zeros((m,2)))                 #用于存放聚类信息的矩阵
    centroid0=np.mean(datamat,axis=0)                      #计算初始的质心坐标
    centerList=[centroid0.tolist()]                                 #存放质心的坐标
    for i in range(m):
        clusterAssment[i,1]=calcEculd(centroid0,datamat[i,:])**2      #计算每个数据到初始质心的距离
    while (len(centerList)<k):
        lowestSSE=float('inf')
        for i in range(len(centerList)):                   #对于第i个质心
            #提取出划分到第i个质心的数据
            ptsInCurrCluster=datamat[np.nonzero(clusterAssment[:,0].A==i)[0],:]
            #对这些数据进行二次划分
            Newcenter,splitClustAss=K_means(ptsInCurrCluster,2)
            #计算二次划分部分划分以后的偏差
            seeSplit=np.sum(splitClustAss[:,1])
            #计算未划分的偏差
            seeNotSplit=np.sum(clusterAssment[np.nonzero(clusterAssment[:,0]!=i)[0],1])
            #比较划分前后的总误差,如果划分后的误差更小则更新数据
            if(seeSplit+seeNotSplit)<lowestSSE:
                bestCentTosplit=i
                bewtNewCents=Newcenter
                bestClutsAss=splitClustAss.copy()
                lowestSSE=seeSplit+seeNotSplit
            #更新划分结果,此次二次划分的得到1类簇归类为最后一类
            bestClutsAss[np.nonzero(bestClutsAss[:,0].A==1)[0],0]=len(centerList)
            #此次二次划分的0类数据归类为划分之前的类
            bestClutsAss[np.nonzero(bestClutsAss[:,0].A==0)[0],0]=bestCentTosplit
            #划分之前的类质心数据更新为二次划分以后的第一个质心
            centerList[bestCentTosplit]=bewtNewCents[0,:]
            #插入二次划分以后新的质心
            centerList.append(bewtNewCents[1,:])
            #更新clusterAssment中的数据
            clusterAssment[np.nonzero(clusterAssment[:,0].A==bestCentTosplit)[0],:]=bestClutsAss
    return centerList,clusterAssment

if __name__=='__main__':
    data,label=load_data()
    orids,Assment=Bin_K_means(data,3)
    print(orids)
