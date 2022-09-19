#回归模型
#比较最小二乘回归与梯度下降回归的效果
import numpy as np
import time
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#产生数据并附带一定的噪声
def load_data(k):
    np.random.seed(0)
    x = np.random.rand(k,1)
    y = 4 + 3 * x+np.random.rand(k,1)
    x1,x2,y1,y2=train_test_split(x,y,test_size=0.4,random_state=42)
    return x1, np.array(y1.T.tolist()[0]), x2, np.array(y2.T.tolist()[0])
if __name__=='__main__':
    x_train,y_train,x_test,y_test=load_data(1000)
    #基于最小二乘训练线性回归模型
    linear_mod=LinearRegression()
    start_1=time.perf_counter()
    linear_mod.fit(x_train,y_train)
    end_1=time.perf_counter()
    y_pred_1=linear_mod.predict(x_test)
    MSE_1=mean_squared_error(y_test,y_pred_1)
    print('最小二乘模型训练时间：%f'%(end_1-start_1))
    print('最小二乘回归模型的均方误差：%f'%(MSE_1))
    #基于梯度下降训练线性回归模型
    SGD_MOD=SGDRegressor(random_state=42)
    start_2 = time.perf_counter()
    SGD_MOD.fit(x_train,y_train)
    end_2 = time.perf_counter()
    y_pred_2=SGD_MOD.predict(x_test)
    MSE_2=mean_squared_error(y_test,y_pred_2)
    print('梯度下降模型训练时间：%f' % (end_2 - start_2))
    print('梯度下降模型的均方误差：%f'%(MSE_2))