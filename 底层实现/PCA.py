#PCA降维
import matplotlib.pyplot as plt
import numpy as np
#随机生成50x3的数据集
def load_data():
    #固定随机数种子，保证运行每次结果都相同
    np.random.seed(0)
    datamat=np.random.rand(30,2)
    return np.mat(datamat)
def PCA(datamat,k=None):
    if k==None:                                   #如果未指定k则返回原数据
        print('未指定降维目标')
        return datamat
    meanmat=np.mean(datamat,axis=0)               #对每一列求平均值
    mean_removed_data=datamat-meanmat             #去中心化
    CovData=np.cov(mean_removed_data,rowvar=False)#计算协方差矩阵，设置参数rowvar，表示列为变量属性
    eigVals,eigVec=np.linalg.eig(CovData)         #计算协方差矩阵的特征值与特征向量
    sorted_eigVal=np.argsort(-eigVals)            #排序并提取前k个特征向量，argsort默认为升序排列，此处需要降序排列
    sorted_eigVal=sorted_eigVal[:k]
    matV=eigVec[:,sorted_eigVal]
    lowD_Data=mean_removed_data*matV              #计算降维以后的数据矩阵
    reconData=lowD_Data*matV.T+meanmat            #将数据转化到N个特征向量的新空间
    return lowD_Data,reconData
'''def PCA_fig(datamat):                             #PCA过程可视化，选取方差较大的两个变量，在二维平面上进行展示
    lowD_Data, sorted_eigVal,mean_removed=PCA(datamat,2)
    main_data=mean_removed[:,sorted_eigVal]
    x=np.array(main_data[:,0])
    y=np.array(main_data[:,1])
    #坐标轴居中
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.spines['top'].set_color('none')            #隐藏多余的坐标轴
    ax.spines['right'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')         #用bottom代替x轴，left代替y轴
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))
    #绘制去中心化以后数据的散点图
    plt.scatter(x,y)
    plt.show()'''
if __name__=='__main__':
    data=load_data()
    lowD_Data,reconData=PCA(data,1)
    plt.scatter(np.array(data[:, 0]), np.array(data[:, 1]), color='blue')
    plt.plot(np.array(reconData[:,0]),np.array(reconData[:,1]),color='red')
    plt.show()
    print(reconData)
