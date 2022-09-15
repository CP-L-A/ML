#回归模型
#线性回归
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
#产生数据并附带一定的噪声
def load_data():
    np.random.seed(0)
    x = np.random.rand(200, 1)
    y = 4 + 3 * x + np.random.rand(200, 1)
    x1 = x[:150, :]
    x2 = x[150:200, :]
    y1 = y[:150, :]
    y2 = y[150:200, :]
    return x1, y1, x2, y2
if __name__=='__main__':
    x_train,y_train,x_test,y_test=load_data()
    #训练线性回归模型
    linear_mod=LinearRegression()
    linear_mod.fit(x_train,y_train)
    y_pred=linear_mod.predict(x_train)
    y_predict=linear_mod.predict(x_test)
    MSE_1=mean_squared_error(y_train,y_pred)
    MSE_2=mean_squared_error(y_test,y_predict)
    print(MSE_1,MSE_2)
    plt.scatter(x_test,y_test,color='red')
    plt.plot(x_test,y_predict)
    plt.show()