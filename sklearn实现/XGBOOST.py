#XGBOOST框架使用练习
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
#数据加载
def load_data():
    data,label=load_iris(return_X_y=True)
    train_d,test_d,train_l,test_l=train_test_split(data,label,test_size=0.3,random_state=42)
    return train_d,train_l,test_d,test_l
#PCA降维，将标准化数据和降维过程封装成为一个pipe
def PCA_Train(datamat,Thresh=0.95):
    from sklearn.preprocessing import StandardScaler
    preprocess=Pipeline([('standard',StandardScaler()),('PCA',PCA(n_components=Thresh,svd_solver='full'))])
    preprocess.fit(datamat)
    lowd_data=preprocess.fit_transform(datamat)
    return lowd_data,preprocess
#训练过程
if __name__=='__main__':
    train_data,train_label,test_data,test_label=load_data()
    lowd_data,transformer=PCA_Train(train_data,0.99)
    #模型训练
    xgb_clf=XGBClassifier(learning_rate=1,max_depth=2,n_estimators=100)
    xgb_clf.fit(lowd_data,train_label)
    #训练集上的性能
    print('训练集上的混淆矩阵:')
    tarin_label_pred=xgb_clf.predict(lowd_data)
    print(confusion_matrix(train_label,tarin_label_pred))
    #测试泛化性能
    print('测试集上的混淆矩阵:')
    lowd_test=transformer.transform(test_data)
    test_label_pred=xgb_clf.predict(lowd_test)
    print(confusion_matrix(test_label,test_label_pred))
