import matplotlib.pyplot as plt
import numpy as np
w=[0,0]
b=0
training_set=[[(1,2),1],[(2,3),1],[(3,1),-1],[(4,2),-1]]
#梯度下降函数，采用随机梯度下降法
def update(item):
    theta=1
    global w,b
    w[0]=w[0]+theta*item[1]*item[0][0]
    w[1]=w[1]+theta*item[1]*item[0][1]
    b=b+theta*item[1]
#判别函数，判断误分类的点
def judge(item):
    s=False
    d=item[1]*(w[0]*item[0][0]+w[1]*item[0][1]+b)
    if d<=0:
       s=True
    return s
#每次遍历所有数据直至所有数据无误分类点
def check(train_1):
    flag=0
    for item in train_1:
        if judge(item):
            update(item)
            flag=1
    if not flag:
       print("w:"+str(w)+"b:"+str(b))
    return flag
#训练过程
def perce():
    #迭代次数为1000次
    for i in range(1000):
        if not check(training_set):break
#绘制示意图
def fig(traindata,w1,w2,b):
    figure=plt.figure()
    ax=figure.add_subplot(111)
    x=np.zeros(len(traindata))
    y=np.zeros(len(traindata))
    i=0
    for item in traindata:
        x[i]=item[0][0]
        y[i]=item[0][1]
        i=i+1
    plt.scatter(x,y,color='red')
    x2=np.linspace(1, 4)
    y2=(-w1/w2)*x2+b
    plt.plot(x2,y2)
    plt.show()
#主程序
def main():
    perce()
    fig(training_set,w[0],w[1],b)
main()