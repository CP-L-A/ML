import math
import operator
#训练数据
train_data=[[1,1,'yes'],[1,1,'yes'],[0,1,'no'],[0,1,'no'],[1,0,'no']]
label=['不浮出水面是否可生存','是否鱼类']
def reloadData():
    traindata=[[1,1,'yes'],[1,1,'yes'],[0,1,'no'],[0,1,'no'],[1,0,'no']]
    trainlabel=['不浮出水面是否可生存','是否鱼类']
    return traindata,trainlabel
#划分数据
def splitData(dataset,index,value):
    retdata=[]                                                              #redata用于存放符合特征的数据
    for item in dataset:
        if item[index]==value:
            retfeature=item[:index]                                         #去除被选中的特征然后加入redata中
            retfeature.extend(item[index+1:])
            retdata.append(retfeature)
    return retdata
#计算香农熵
def calcShannonEnt(dataset):
    num=len(dataset)
    LabelCounts={}
    for item in dataset:
        label=item[-1]
        LabelCounts[label]=LabelCounts.get(label,0)+1
    ShannonEnt=0                                                            #默认香农熵为0
    for key in LabelCounts:
        p=LabelCounts[key]/num
        ShannonEnt-=p*math.log(p,2)                                         #香农熵计算公式H=SUM(-P*logP)
    return ShannonEnt
#选择最优划分特征
def ChooseFeature(train_data):
    featurenum=len(train_data[0])-1
    #基础香农熵
    baseEnt=calcShannonEnt(train_data)
    bestInfoGain=0
    bestFeature=0
    for i in range(featurenum):
       #提取数据列表中的特征
       featurelist=[example[i] for example in train_data]
       #设置数据为集合类型，避免出现重复值
       Vals=set(featurelist)
       newEnt=0
       #计算条件香农熵，公式为H=P1*H(D1)+P2*H(D2)
       for value in Vals:
           subData=splitData(train_data,i,value)
           prob=len(subData)/len(train_data)
           newEnt+=prob*calcShannonEnt(subData)
       #计算信息增益
       newInfoGain=baseEnt-newEnt
       #如果该特征信息增益大于最优信息增益，则选取该特征为最优特征
       if newInfoGain>bestInfoGain:
           bestInfoGain=newInfoGain
           bestFeature=i
    return bestFeature
#投票表决
def majorcount(classlist):
    classCount={}
    for key in classlist:
        classCount[key]=classCount.get(key,0)+1
    sortedcount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedcount[0][0]
#递归构造决策树，基于字典构造
def creatTree(dataset,labels):
    classlist=[example[-1] for example in dataset]
    #当前数据同属一个类别时停止递归
    if classlist.count(classlist[0])==len(classlist):
        return classlist[0]
    #若已经使用所有特征当前数据仍为不同类，则返回出现次数最多的类别
    if len(dataset[0])==1:
        return majorcount(classlist)
    #确定最优的划分特征
    bestFeat=ChooseFeature(dataset)
    #划分特征对应的实际特征
    bestlabel=labels[bestFeat]
    #建立空的字典
    Tree={bestlabel:{}}
    #删除该标签，避免重复出现
    del(labels[bestFeat])
    #提取特征
    featValue=[example[bestFeat] for example in dataset]
    uniqueVals=set(featValue)
    for value in uniqueVals:
        sublabels=labels[:]
        Tree[bestlabel][value]=creatTree(splitData(dataset,bestFeat,value), sublabels)
    return Tree  
#查找决策树
def classify(test_data,labels,Tree):
    #提取当前特征
    firstStr=list(Tree.keys())[0]
    #当前特征的划分结果
    subDict=Tree[firstStr]
    #确定当前特征对于测试数据的下标
    featIndex=labels.index(firstStr)
    #判断测试数据在当前特征下的结果
    for key in subDict.keys():
        if test_data[featIndex]==key:
            #如果仍可继续划分，则继续进行分类
            if type(subDict[key]).__name__=='dict':
                classlabel=classify(test_data,labels,subDict[key])
            else:
            #若当前特征不可划分，则返回当前类别
                classlabel=subDict[key]
    return classlabel
#主函数
def main():
    data,label=reloadData()
    Tree=creatTree(data, label)
    print(Tree)
    #由于label标签已经修改，需要重新加载数据
    data,label=reloadData()
    classedlabel=classify([0,1], label, Tree)
    print(classedlabel)
main()