from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import time
import numpy as np
def data_load():
    iris=load_iris()
    train_data,test_data,train_label,test_label=train_test_split(iris.data,iris.target,
                                                                 test_size=0.4,
                                                                 random_state=42)

    return train_data,train_label,test_data,test_label

def GBDT(data,label,num=20,depth=1):
    gradTree=GradientBoostingClassifier(n_estimators=num,
                                        learning_rate=1.0,
                                        max_depth=depth,
                                        random_state=42)
    gradTree.fit(data,label)
    return gradTree
if __name__=='__main__':
    datamat,labelmat,testdata,testlabel=data_load()
    start=time.perf_counter()
    Tree=GBDT(datamat,labelmat,100)
    end=time.perf_counter()
    print('运行时间:'+str(end-start))
    result=Tree.predict(testdata)
    print(result)
    print(testlabel)