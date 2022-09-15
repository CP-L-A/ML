#随机梯度下降的sk实现
import os
import struct
import numpy as np
import time
import joblib
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
train_image_path=r'C:\Users\LCP\Desktop\MNIST\train-images.idx3-ubyte'
label_path=r'C:\Users\LCP\Desktop\MNIST\train-labels.idx1-ubyte'
#加载mnist数据
with open(label_path,'rb') as lbpath:
    magic,n=struct.unpack('>II',lbpath.read(8))
    labels=np.fromfile(lbpath,dtype=np.uint8)
with open(train_image_path,'rb') as trainpath:
    magic,num,rows,cols=struct.unpack('>IIII',trainpath.read(16))
    images=np.fromfile(trainpath,dtype=np.uint8).reshape(len(labels),784)
y_train=(labels==3)
start=time.perf_counter()
SGD_MOD=SGDClassifier(random_state=42)
SGD_MOD.fit(images,y_train)
end=time.perf_counter()
print('运行耗时',end-start)
joblib.dump(SGD_MOD,'save/SGD_MOD.pkl')                                #保存模型

