#基于单层决策树的Adaboost算法实现
import numpy as np
import math
#数据加载
def load_data():
    dataMat=[[1,2],
             [2,1.1],
             [1.3,1],
             [1,1],
             [2,1]]
    labels=[1,1,-1,-1,1]
    return np.mat(dataMat),np.mat(labels).T
#构造单层决策树，输入参数包括原数据集，特征索引，阈值以及分类方式
def stumpClassify(datamat,dimen,threshVal,threshIneq):
    retArry=np.ones((np.shape(datamat)[0],1))
    if threshIneq=='lt':
        retArry[datamat[:,dimen]<=threshVal]=-1
    else:
        retArry[datamat[:,dimen]>threshVal]=-1
    return retArry
#基于当前权重构建最优决策树
#根据在最优划分特征，划分阈值及划分方式
def bulidstump(datamat,labelmat,D):
    m,n=np.shape(datamat)
    numsteps=10
    beststump={}
    bestClsEst=np.mat(np.zeros((m,1)))
    minErr=float('inf')
    for i in range(n):
        rangeMin=datamat[:,i].min()
        rangeMax=datamat[:,i].max()
        stepsize=(rangeMax-rangeMin)/numsteps                       #计算步长
        for j in range(-1,int(numsteps)+1):
            for inequal in ['lt','gt']:
                threshVal=(rangeMin+float(j)*stepsize)              #根据步长计算阈值
                predict=stumpClassify(datamat,i,threshVal,inequal)  #返回当前阈值条件下的分类结果
                errArr=np.mat(np.ones((m,1)))
                errArr[predict==labelmat]=0                         #计算每个数据的误差
                weightErr=D.T*errArr                                #计算加权误差
                if weightErr<minErr:
                    minErr=weightErr
                    bestClsEst=predict.copy()
                    beststump['dim']=i
                    beststump['threshVal']=threshVal
                    beststump['threshIng']=inequal
    return beststump,float(minErr),bestClsEst                       #返回最优决策方式，包括最优特征，阈值以及分类方式
#AdaBoost训练过程，默认最大迭代次数为100
def AdaBoost(datamat,label,maxIt=100):
    weckArr=[]                                                      #建立一个列表用于存放最终分类结果
    m=np.shape(datamat)[0]
    D=np.mat(np.ones((m,1))/m)                                      #初始权重，取平均值
    aggClassE=np.mat(np.zeros((m,1)))                               #建立一个矩阵以存放总的分数
    for i in range(maxIt):                                          #进入迭代
        beststump,minErr,bestClsEst=bulidstump(datamat,label,D)
        alpha=float(0.5*math.log((1-minErr)/max(minErr,1e-16)))     #计算alpha值
        beststump['alpha']=alpha
        weckArr.append(beststump)
        expon=-1*alpha*np.multiply(label,bestClsEst)                #计算新一轮的权重，计算公式参考《李航统计学习》
        D=np.multiply(D,np.exp(expon))
        Z=D.sum()
        D=D/Z
        aggClassE+=alpha*bestClsEst                                 #计算总的得分
        aggError=np.multiply(np.sign(aggClassE)!=label,np.ones((m,1)))   #判断误分类点个数及误分类率
        errRate=aggError.sum()/m
        if errRate==0:break                                         #没有误分类点时则退出循环
    return weckArr
def BoostClassify(datamat,weckArr):
    datamat=np.mat(datamat)
    m=np.shape(datamat)[0]
    score=np.zeros((m,1))
    for i in range(len(weckArr)):
        print(weckArr[i]['dim'],weckArr[i]['threshVal'],weckArr[i]['threshIng'])
        predict=stumpClassify(datamat,weckArr[i]['dim'],
                              weckArr[i]['threshVal'],
                              weckArr[i]['threshIng'])
        score+=weckArr[i]['alpha']*predict
    return np.sign(score).T
if __name__=='__main__':
    datamat,labelmat=load_data()
    Gx=AdaBoost(datamat,labelmat,50)
    print(Gx)
    print(BoostClassify(datamat,Gx))