#KNN基于sklearn的实现，并采用了PCA降维
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
def load_data():
    data,label=load_iris(return_X_y=True)
    train_data,test_data,train_label,test_label=train_test_split(data,label,test_size=0.3,random_state=42)
    return train_data,train_label,test_data,test_label
#协方差分析选择合适的k
def Var_analysis(datamat,Thresh=0.95):
    covdata=np.cov(datamat,rowvar=False)
    eigVal,eigVec=np.linalg.eig(covdata)
    eigVal=np.sort(eigVal)[::-1]
    sum_1=0
    sum_2=np.sum(eigVal)
    for i in range(len(eigVal)):
        sum_1+=eigVal[i]
        if sum_1/sum_2 >=Thresh:
            return i
#KNN模型训练
def KNNClassify(datamat,label,k=3):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(datamat,label)
    return knn
#根据分析结果进行降维
def PCA_(data,K):
    PCA_MOD=PCA(n_components=K)
    PCA_MOD.fit(data)
    return PCA_MOD.transform(data)

if __name__=='__main__':
    train_data,train_label,test_data,test_label=load_data()
    k=Var_analysis(train_data,0.99)
    lowd_data=PCA_(train_data,k)
    lowd_test=PCA_(test_data,k)
    knn=KNNClassify(lowd_data,train_label)
    label_pred=knn.predict(lowd_test)
    print('分类结果:\n',label_pred)

