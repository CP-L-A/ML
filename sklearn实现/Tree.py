import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
import time
import joblib
#加载鸢尾花数据集
def data_load():
    iris=load_iris()
    datamat=iris.data
    labelmat=iris.target
    return datamat,labelmat,iris
#树训练过程
def Tree(data,label):
    tree=DecisionTreeClassifier(random_state=42)
    start = time.perf_counter()
    tree.fit(data,label)
    end = time.perf_counter()
    print("运行时间:"+str(end-start)+"秒")
    return tree
def test_load(iris):
    train_data,test_data,train_label,test_label=train_test_split(iris.data,iris.target,test_size=0.4)
    return test_data,test_label
#保存树文件
def save_tree(tree):
    joblib.dump(tree,'save/Tree.pkl')
if __name__=='__main__':
    x,y,iris=data_load()
    y_train=(y==2)
    test_data,test_label=test_load(iris)
    test_label=(test_label==2)
    tree_classify=Tree(x,y_train)
    pred=tree_classify.predict(test_data)
    plot_tree(tree_classify)
    recall=recall_score(test_label,pred)
    print(recall)
    plt.show()