#梯度提升树的sk实现
#建立梯度提升回归树拟合线性模型
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
def load_data():
    np.random.seed(0)
    x=np.random.rand(200,1)
    y=4+3*x*x+np.random.rand(200,1)
    x1=x[:150,:]
    x2=x[150:200,:]
    y1=y[:150,:]
    y2=y[150:200,:]
    return x1, np.array(y1.T.tolist()[0]), x2, np.array(y2.T.tolist()[0])
#GBDT的训练过程
def GBDT_Train(x,y):
    reg_tree=GradientBoostingRegressor(n_estimators=150,max_depth=1)
    reg_tree.fit(x,y)
    return reg_tree
if __name__=='__main__':
    x_train,y_train,x_test,y_test=load_data()
    tree=GBDT_Train(x_train,y_train)
    y_pred=tree.predict(x_test)
    y_prec=tree.predict(x_train)
    #计算均方误差
    mean_error_1=mean_squared_error(y_train,y_prec)         #训练集上的均方误差
    mean_error_2=mean_squared_error(y_test,y_pred)          #测试集的均方误差
    print('训练集均方误差：%f,测试集均方误差：%f'%(mean_error_1, mean_error_2))




