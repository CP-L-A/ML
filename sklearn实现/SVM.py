#基于sklearn的SVM分类器实现实现
#采用sklearn自带的乳腺癌数据集
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import joblib
#标志变量：决定调用已有模型或新的模型
MOD_load=False
def load_model():
    MOD=joblib.load('save/SVM.pkl')
    return MOD
#数据集加载，并划分数据集
def load_data():
    data,label=load_breast_cancer(return_X_y=True)
    train_data,test_data,train_label,test_label=train_test_split(data,label,test_size=0.3,random_state=42)
    return data,label,test_data,test_label
#分析数据集的有效信息，为降维提供参考
def Var_analysis(data,Thresh=0.95):
    covdata=np.cov(data, rowvar=False)                   #计算协方差矩阵
    eigVal,eigVec=np.linalg.eig(covdata)                 #求解协方差矩阵特征值
    eigVal=np.sort(eigVal)[::-1]                         #降序排列
    sum=0
    for i in range(len(data)):                           #计算达到设定阈值所需要的最小特征数目（阈值默认为0.95）
        sum+=eigVal[i]
        if sum/(np.sum(eigVal))>=Thresh:
            return i+1
#PCA降维过程
def PCA_Train(data,k):
    PCA_MOD=PCA(n_components=k)
    PCA_MOD.fit(data)
    lowD_Data=PCA_MOD.transform(data)
    return lowD_Data
#SVM训练过程
def SVM_Train(x,y):
    model=svm.SVC(kernel='rbf',C=0.1)                    #核函数采用rbf
    model.fit(x,y)
    return model
#保存SVM模型
def save_mod(MOD):
    joblib.dump(MOD,'save/SVM.pkl')
if __name__=='__main__':
    data,label,test_d,test_l=load_data()
    k=Var_analysis(data,0.99)                           #分析有效信息并进行降维
    lowd_data=PCA_Train(data,k)
    if MOD_load:
        SVM_MOD=load_model()
    else:
        SVM_MOD=SVM_Train(lowd_data,label)
        #save_mod(SVM_MOD)
    #在训练集上进行测试
    lowd_test=PCA_Train(test_d,k)
    pred_label=SVM_MOD.predict(lowd_test)
    #计算精度和召回率
    train_precision=precision_score(label,SVM_MOD.predict(lowd_data))
    print(train_precision)
    precision=precision_score(test_l,pred_label)
    recall=recall_score(test_l,pred_label)
    f1=f1_score(test_l,pred_label)
    print('精度:%f ,召回率:%f ,F1:%f'%(precision*100,recall*100,f1*100))
