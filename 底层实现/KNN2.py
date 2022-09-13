#本文件为直接遍历所有数据，计算距离的KNN实现方法
import numpy as np
import math
import operator
#训练数据
group=np.array([[1,1.1],[1,1],[0,0],[0,0.1]])
Distance=[]
labels=['A','A','B','B']
#计算距离
def cal(item1,item2):
    if len(item1)!=len(item2):
        #如果两组数据长度不一致则说明数据有问题
        print("数据长度有误！")
        return 0
    else:
        d2=0
        for i in range(len(item1)):
            d2=d2+math.pow((item1[i]-item2[i]),2)
        d=math.sqrt(d2)
    return d

def KNN(inX,dataset,labels,k):
    global Distance
    length=len(dataset)
    #遍历所有数据，计算距离
    for i in range(length):
        distance=cal(inX,dataset[i])
        Distance.append(distance)  
    #列表转数组，进行排序
    Disarray=np.array(Distance)
    #排序，并将排序后的数据的下标按顺序存放进新数组
    sortedistance=Disarray.argsort()
    #建立字典以存放各类型的数据所对应的近邻点个数
    classCount={}
    for i in range(k):
        votelabel=labels[sortedistance[i]]
        classCount[votelabel]=classCount.get(votelabel,0)+1
    #降序排列，排序规则按照符合要求元素的个数排列
    #第一个参数表表明对classCount的元素进行排列，第二个参数代表按照个数进行排列，第三个参数代表降序排列
    sortedClasscount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    #选择个数最多的那个类型并返回
    return sortedClasscount[0][0]

def main():
    classify=KNN([1,2],group,labels,1)
    print("待测样本分类:"+str(classify))

main()