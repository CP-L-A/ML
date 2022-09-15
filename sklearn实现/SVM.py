#基于sklearn的SVM实现
from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import joblib
def load_data():
    iris=load_iris()
    x=iris.data
    y=iris.target
    x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.4,random_state=42)
    y_train=(y_train==0)
    y_test=(y_test==0)
    return x_train,y_train,x_test,y_test
def SVM_Train(x,y):
    model=svm.SVC(kernel='linear')
    model.fit(x,y)
    return model
def save_mod(MOD):
    joblib.dump(MOD,'save/SVM.pkl')
if __name__=='__main__':
    train_data,train_label,test_data,test_label=load_data()
    SVM_MOD=SVM_Train(train_data,train_label)
    y_pred=SVM_MOD.predict(test_data)
    save_mod(SVM_MOD)
    score=recall_score(test_label,y_pred)
    print(score)
    print(SVM_MOD.coef_)