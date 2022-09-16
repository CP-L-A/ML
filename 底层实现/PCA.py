#PCA降维
import matplotlib.pyplot as plt
import numpy as np
def load_data():
    #固定随机数种子
    np.random.seed(0)
    datamat=np.random.rand(50,3)
    return np.mat(datamat)
def PCA(datamat,k=None):
    if k==None:                                   #如果未指定k则返回原数据
        print('未指定降维目标')
        return datamat
    meanmat=np.mean(datamat,axis=0)
    mean_removed_data=datamat-meanmat             #去中心化
    CovData=np.cov(mean_removed_data,rowvar=None) #计算协方差矩阵
    eigVals,eigVec=np.linalg.eig(CovData)         #计算协方差矩阵的特征值与特征向量
    sorted_eigVal=np.argsort(-eigVals)            #排序并提取前k个特征向量，argsort默认为升序排列，此处需要降序排列
    sorted_eigVal=sorted_eigVal[:k]
    matV=eigVec[:,sorted_eigVal]
    lowD_Data=mean_removed_data*matV              #计算降维以后的矩阵
    return lowD_Data
if __name__=='__main__':
    data=load_data()
    PCA_Data=PCA(data,2)
