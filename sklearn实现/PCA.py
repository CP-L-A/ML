#PCA的sklearn实现
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
def load_data():
    np.random.seed(0)
    data=np.random.rand(50,3)
    return data
def PCA_Train(data,k=None):
    PCA_MOD=PCA(n_components=k)
    PCA_MOD.fit(data)
    return PCA_MOD
if __name__=='__main__':
    datamat=load_data()
    PCA_MOD=PCA_Train(datamat,2)
    lowD_Data=PCA_MOD.transform(datamat)
    reconData=PCA_MOD.inverse_transform(lowD_Data)
    # 设置绘制三维图
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # 原始数据的散点图
    ax.scatter3D(np.array(datamat[:, 0]), np.array(datamat[:, 1]), np.array(datamat[:, 2]), color='blue')
    ax.scatter3D(np.array(reconData[:, 0]), np.array(reconData[:, 1]), np.array(reconData[:, 2]), color='red')
    plt.show()
