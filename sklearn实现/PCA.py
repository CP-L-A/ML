#PCA的sklearn实现
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.linear_model import LinearRegression
import numpy as np
def load_data():
    np.random.seed(0)
    data=np.random.rand(50,3)
    return data
def PCA_Train(data,k=None):
    PCA_MOD=PCA(n_components=k)
    PCA_MOD.fit(data)
    return PCA_MOD
def pic(data):
    train_data=data[:,:2]
    train_z=data[:,-1]
    liner_mod=LinearRegression()
    liner_mod.fit(train_data,train_z)
    return liner_mod
if __name__=='__main__':
    data = load_data()
    PCA_MOD=PCA_Train(data,2)
    lowD_Data=PCA_MOD.transform(data)
    reconData=PCA_MOD.inverse_transform(lowD_Data)
    # 设置绘制三维图
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # 原始数据的散点图
    ax.scatter3D(np.array(data[:, 0]), np.array(data[:, 1]), np.array(data[:, 2]), color='blue')
    # 绘制平面
    linear = pic(reconData)
    w = linear.coef_.tolist()
    w0 = linear.intercept_
    x = np.linspace(-0.1, 1, 10)
    y = np.linspace(-0.1, 1, 10)
    xx, yy = np.meshgrid(x, y)
    print(w)
    zz = float(w0) + float(w[0]) * xx + float(w[1]) * yy
    # ax.plot_wireframe(xx,yy,zz,color='red')
    ax.plot_surface(xx, yy, zz, color='skyblue', alpha=0.3)
    # 绘制降维以后的数据点
    ax.scatter3D(np.array(reconData[:, 0]), np.array(reconData[:, 1]), np.array(reconData[:, 2]), color='red')
    # 将降维以后的数据点和原数据连接起来
    for i in range(len(data)):
        ax.plot([np.array(data[i, 0]), np.array(reconData[i, 0])],
                [np.array(data[i, 1]), np.array(reconData[i, 1])],
                [np.array(data[i, 2]), np.array(reconData[i, 2])], color='red')
    plt.show()