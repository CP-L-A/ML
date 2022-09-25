#logistic回归的sklearn实现
#使用K均值聚类进行降维
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.pipeline import  Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from nyoka import skl_to_pmml
#数据加载
def load_data():
    data,label=load_iris(return_X_y=True)
    feature_name=load_iris().feature_names
    return data,label,feature_name
#主函数
if __name__=='__main__':
    datamat,labelmat,features=load_data()
    print(features)
    #将k均值聚类和逻辑斯蒂回归封装为一个管道
    log_pipe=Pipeline([('mean',KMeans(n_clusters=75)),
                       ('log_reg',LogisticRegression(multi_class="multinomial",solver='lbfgs',max_iter=1000))])
    log_pipe.fit(datamat,labelmat)
    label_pred=log_pipe.predict(datamat)
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(labelmat,label_pred))
