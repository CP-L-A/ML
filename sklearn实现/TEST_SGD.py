import os
import struct
import numpy as np
import time
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
#加载之前保存好的模型
SGD=joblib.load('save/SGD_MOD.pkl')
def load_test():
    test_label_path = r'C:\Users\LCP\Desktop\MNIST\t10k-labels.idx1-ubyte'
    with open(test_label_path,'rb') as testlabel:
        magic, n = struct.unpack('>II', testlabel.read(8))
        labels = np.fromfile(testlabel, dtype=np.uint8)
    test_image_path = r'C:\Users\LCP\Desktop\MNIST\t10k-images.idx3-ubyte'
    with open(test_image_path,'rb') as testimage:
        magic, num, rows, cols = struct.unpack('>IIII', testimage.read(16))
        images = np.fromfile(testimage, dtype=np.uint8).reshape(len(labels), 784)
    return images,labels
x_test,y_test=load_test()
y_test_3=(y_test==3)
x_label=SGD.predict(x_test)
en=(x_label==y_test_3)
dic={}
for item in en:
    dic[item]=dic.get(item,0)+1
print(dic[True]/len(x_label))
