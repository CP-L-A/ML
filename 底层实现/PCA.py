#PCA降维
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.animation as anim
from sklearn.linear_model import LinearRegression
import numpy as np
import time
#随机生成50x3的数据集
def load_data():
    #固定随机数种子，保证运行每次结果都相同
    np.random.seed(42)
    datamat=np.random.rand(50,3)
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
#线性回归，拟合降维之后的平面
def pic(data):
    train_data=data[:,:2]
    train_z=data[:,-1]
    liner_mod=LinearRegression()
    liner_mod.fit(train_data,train_z)
    return liner_mod
if __name__=='__main__':
    data=load_data()
    lowD_Data,reconData=PCA(data,2)
    #计算降维后数据所在的平面
    linear= pic(reconData)
    w=linear.coef_.tolist()
    w0=linear.intercept_
    x=np.linspace(0,1,5)
    y=np.linspace(0,1,5)
    xx,yy=np.meshgrid(x,y)
    zz = float(w0) + float(w[0][0]) * xx + float(w[0][1]) * yy
    # 设置绘制三维图
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_zlim(-1,2)
    plt.ion()
    # 原始数据的散点图
    ax.scatter3D(data[:, 0],data[:, 1], data[:,2], color='blue')
    plt.pause(12)
    # 绘制平面
    ax.plot_surface(xx, yy, zz, color='green', alpha=0.3)
    plt.pause(2)
    # 将降维以后的数据点和原数据连接起来
    for i in range(len(data)):
        ax.plot([np.array(data[i, 0]), np.array(reconData[i, 0])],
                [np.array(data[i, 1]), np.array(reconData[i, 1])],
                [np.array(data[i, 2]), np.array(reconData[i, 2])], color='black',linewidth=0.5)
    plt.pause(0.1)
    # 绘制降维以后的数据点
    ax.scatter3D(np.array(reconData[:, 0]), np.array(reconData[:, 1]), np.array(reconData[:, 2]), color='red')
    plt.pause(30)
    plt.ioff()