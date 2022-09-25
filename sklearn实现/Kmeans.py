#K均值聚类的sklearn实现,数据集采用鸢尾花数据集
#原数据集默认为3类
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn import pipeline
from nyoka import skl_to_pmml

#加载数据集
def load_data():
    data,label=load_iris(return_X_y=True)
    return data

#标准化数据
def Standard(data,Thresh=0.9):
    stand=StandardScaler()
    stand.fit(data)
    scaled_data=stand.transform(data)
    return scaled_data,stand

#方差分析
def SVD_Analysis(data,Thresh):
    U,Sigmod,D=np.linalg.svd(data)
    Sig=np.sort(Sigmod)[::-1]
    k=0
    SumSig=np.sum(Sig)
    for i in range(len(Sig)):
        k+=Sig[i]
        if k/SumSig>=Thresh:
            return i

if __name__=='__main__':
    data=load_data()
    k=SVD_Analysis(data,0.99)
    Pipe=pipeline.Pipeline([('scl',StandardScaler()),
                            ('PCA',PCA(n_components=k)),
                            ('Kmean',KMeans(n_clusters=3))])
    Pipe.fit(data)



