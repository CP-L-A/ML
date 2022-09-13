import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
import time
import joblib
def data_load():
    iris=load_iris()
    datamat=iris.data
    labelmat=iris.target
    return datamat,labelmat,iris
def Tree(data,label):
    tree=DecisionTreeClassifier(random_state=42)
    start = time.perf_counter()
    tree.fit(data,label)
    end = time.perf_counter()
    print("运行时间:"+str(end-start)+"秒")
    return tree
def save_tree(tree):
    joblib.dump(tree,'save/Tree.pkl')
if __name__=='__main__':
    x,y,iris=data_load()
    train_data,test_data=train_test_split(x,test_size=0.4,random_state=42)
    train_label,test_label=train_test_split(y,test_size=0.4,random_state=42)
    tree_classify=Tree(train_data,train_label)
    predict_label=tree_classify.predict(test_data)
    save_tree(tree_classify)
    plot_tree(tree_classify)
    plt.show()