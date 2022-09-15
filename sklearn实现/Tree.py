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
    datamat,test_data,labelmat,test_label=train_test_split(iris.data,iris.target,test_size=0.4,random_state=42)
    return datamat,labelmat,test_data,test_label
#树训练过程
def Tree(data,label):
    tree=DecisionTreeClassifier(random_state=42)
    tree.fit(data,label)
    return tree
#保存树文件
def save_tree(tree):
    joblib.dump(tree,'save/Tree.pkl')
if __name__=='__main__':
    train_data,train_label,test_data,test_label=data_load()
    classifier=Tree(train_data,train_label)
    pred=classifier.predict(test_data)
    save_tree(classifier)
    bol=(pred==test_label)
    plot_tree(classifier)                                    #决策树可视化
    plt.show()
    x=sum(bol)
    print(x/len(bol))

