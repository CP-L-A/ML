#朴素贝叶斯的底层实现
import numpy as np
import pandas as pd
def load_data():
    datamat=pd.DataFrame({'X1':[1]*5+[2]*5+[3]*5,
                          'X2':['S']+['M']*2+['S']*3+['M']*2+['L']*3+['M']*2+['L']*2,
                          'Y':[-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1]})
    return datamat

class Bayes:
    def __init__(self):
        self.model={}
    #模型拟合，具体学习过程需要buildBayes()函数
    def fit(self,datamat,label=None):
        #如果分类标签与数据是分开的，则进行合并,若没有传入标签，默认数据集最后一列为分类标签
        if label is not None:
            datamat=pd.concat([datamat,label],axis=1)
        self.model=self.buildBayes(datamat)
        return self.model
    #模型的学习过程
    def buildBayes(self,datamat):          #此函数用于学习数据的潜在概率分布
        label=datamat.iloc[:,-1]
        labelcounts=label.value_counts()         #统计各个标签出现的频次
        #拉普拉斯平滑计算先验概率
        #lambda x快速定义了一个x+1的函数并应用于一列数据
        labelcounts=labelcounts.apply(lambda x:(x+1)/(datamat.iloc[:,-1].size+labelcounts.size))
        #建立一个字典存放先验概率和条件概率
        retModel={}
        for label,P in labelcounts.items():
            retModel[label]={'PClass':P,'PFeature':{}}
        Feature=datamat.columns[:-1]
        PropByFeature={}
        for item in Feature:
            #提取样本中每个属性的可能取值
            PropByFeature[item]=list(datamat[item].value_counts().index)
        #将datamat中的数据按照分类标签（数据集的最后一列）进行分组
        for nameClass,group in datamat.groupby(datamat.columns[-1]):
            #对于每一个组，遍历所有特征
            for nameFeature in Feature:
                eachClassP={}
                propDatas=group[nameFeature]
                propClassSum=propDatas.value_counts()
                #计算Ni用于拉普拉斯平滑
                Ni=len(PropByFeature[nameFeature])
                propClassSum=propClassSum.apply(lambda x:(x+1)/(propDatas.size+Ni))
                for nameFeatureProp,valP in propClassSum.items():
                    eachClassP[nameFeatureProp]=valP
                retModel[nameClass]['PFeature'][nameFeature]=eachClassP
        return retModel
    #预测函数，对于每个测试数据，进行预测
    def predict(self,data):
        datamat=pd.DataFrame(data)
        label_pred=[]
        for i in range(len(data)):
            label_pred.append(self.predictBayes(datamat.iloc[i]))
        return label_pred
    #计算条件概率的函数，根据上述已学习得到的模型计算条件概率
    def predictBayes(self,datamat):
        curMaxP=None
        curMaxSelect=None
        for nameClass,infoModel in self.model.items():
            Prob=0
            Prob+=np.log(infoModel['PClass'])
            #查询当前分类标签下的各个条件概率
            PFeature=infoModel['PFeature']
            for feat,vals in datamat.items():
                print('当前所查询的测试数据特征：',feat)
                print('特征的取值',vals)
                ProbRate=PFeature[feat]
                Prob+=np.log(ProbRate.get(vals))
            print('该类别下的后验概率:',Prob)
            if curMaxP==None or Prob>curMaxP:
                curMaxP=Prob
                curMaxSelect=nameClass
        return curMaxSelect

if __name__=='__main__':
    datamat=load_data()
    bayes=Bayes()
    bayes.fit(datamat)
    test_data=pd.DataFrame({'X1':[2],'X2':['S']})
    print(test_data)
    print(bayes.predict(test_data))






