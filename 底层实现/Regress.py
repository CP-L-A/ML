#线性回归问题实现
import matplotlib.pyplot as plt
import numpy as np
#数据加载，随机产生并有一定的噪声
def load_data():
    np.random.seed(0)
    x=np.random.rand(200,1)
    y=4+3*x+np.random.rand(200,1)
    x1=x[:150,:]                      #划分数据集，取前150个元素用于拟合，后50个数据测试
    y1=y[:150,:]
    x2=x[150:200,:]
    y2=y[150:200,:]
    return np.mat(x1),np.mat(y1),np.mat(x2),np.mat(y2)
#数据x的处理，末尾加1,用于计算截距
def x_append_1(x):
    x = x.tolist()
    for i in range(len(x)):
        x[i].append(1)
    return np.mat(x)
#基于最小二乘法的回归，计算公式为w=(xTx)^-1*x.T*y
def Regress_train(x,y):
    xMat=x_append_1(x)
    yMat=np.mat(y)
    xTx=xMat.T*xMat
    if np.linalg.det(xTx)==0:
        print("xTx矩阵不可逆")
        return
    else:
        w=xTx.I*(xMat.T*yMat)
    return w
def cale_MSE(y_true,y_pred):
    error=y_true-y_pred
    MSE=error.T*error
    return float(MSE)/len(error)
#主函数
if __name__=='__main__':
    x_train,y_train,x_test,y_test=load_data()
    w=Regress_train(x_train,y_train)                    #训练模型
    xMat=x_append_1(x_test)                             #进行预测
    y_pred=w.T*xMat.T
    MSE=cale_MSE(y_test,y_pred.T)
    print(MSE)
    plt.scatter(x_test.tolist(),y_test.tolist())        #绘制示意图
    plt.plot(x_test.tolist(),y_pred.T.tolist(),color='red')
    plt.show()