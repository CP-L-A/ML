#回归模型
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
def load_data():
    x=np.random.rand(100,1)
    y=4+3*x+np.random.rand(100,1)
    return x,y
if __name__=='__main__':
    x_train,y_train=load_data()
    plt.scatter(x_train,y_train)
    linear_mod=LinearRegression()
    linear_mod.fit(x_train,y_train)
    b=linear_mod.intercept_
    w=linear_mod.coef_
    y_predict=linear_mod.predict(x_train)
    plt.plot(x_train,y_predict)
    plt.show()