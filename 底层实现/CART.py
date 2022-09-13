#CART算法实现
import numpy as np
#数据集划分函数，参数为数据集，给定特征和阈值
def spilit(dataset,feat,value):
    mat0=dataset[np.nonzero(dataset[:,feat]<=value)[0],:]               #提取出特征所在列，并与value比较，进而生成布尔矩阵
    mat1=dataset[np.nonzero(dataset[:,feat]>value)[0],:]                #通过nonzero函数得到满足条件的数据所在的行
    return mat0,mat1
#计算叶节点的返回值，即各个变量的均值
def regleaf(dataset):
    return np.mean(dataset[:,-1])
#计算总的方差
def regErr(dataset):
    return np.var(dataset[:,-1])*np.shape(dataset)[0]
#选择最优划分特征及划分点
def Bestfeat(dataset,leafType=regleaf,errType=regErr,ops=(1,4)):           
    if len(set(dataset[:,-1].T.tolist()))==1:                         #提取数据集最后一列的数据并转为集合，以此判断当前数据集是否全为一个值
        return None,leafType(dataset)                                 #如果全为一个值，则返回平均值
    m,n=np.shape(dataset)                                              
    S=errType(dataset)                                                #计算原本的方差
    bestS=float('inf')                                                #设原先方差为无穷大
    bestIndex=0                                               
    bestvalue=0
    for featIndex in range(n-1):                                      #遍历所有的特征，最后一列为函数值
        for spilitVal in set(dataset[:,featIndex]):                   #遍历当前特征下所有可能的切分点
            mat0,mat1=spilit(dataset, featIndex, spilitVal)           #按照当前切分点分隔数据
            if np.shape(mat0)==0 or np.shape(mat1)==0:                #如果分隔出有一个数据集无数据则继续遍历
                continue
            newS=errType(mat0)+errType(mat1)                          #计算新的方差，并寻找最小方差
            if newS<bestS:
                bestIndex=featIndex
                bestvalue=spilitVal
                bestS=newS
    if (S-bestS)<1:                                                  #如果方差的变化很小，则不进行分隔，直接返回当前数据集的均值
        return None,leafType(dataset)
    if np.shape(mat0)==0 or np.shape(mat1)==0:
        return None,leafType(dataset)
    return bestIndex,bestvalue
#递归创建回归树
def creatTree(dataset,leafType=regleaf,errType=regErr,ops=(1,4)):   #递归创建回归树
    feat,Val=Bestfeat(dataset,leafType,errType,ops)
    if feat==None:
        return Val
    retTree={}
    retTree['spInd']=feat
    retTree['spVal']=Val
    lSet,rSet=spilit(dataset, feat, Val)
    retTree['left']=creatTree(lSet,leafType,errType,ops)
    retTree['right']=creatTree(rSet,leafType,errType,ops)
    return retTree
if __name__=='__main__':
    s=[[1,2],[3,4],[5,6],[7,8],[8,9]]
    s=np.array(s)
    Tree=creatTree(s)
    print(Tree)
    