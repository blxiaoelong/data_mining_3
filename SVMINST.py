import os
import struct
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
from sklearn import  svm
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
import sklearn.metrics as sm

path = ['train-labels.idx1-ubyte','train-images.idx3-ubyte',
        't10k-labels.idx1-ubyte','t10k-images.idx3-ubyte']



def Getdata(labels_path, images_path):
    transformer = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8)) 
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16)) 
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    print(images.shape)
    images = transformer(images)
    return images.squeeze(0), labels

def train():#train_num为训练的量，最多6个W
    x, y = Getdata(path[0],path[1])  # 加载训练集
    #X = preprocessing.StandardScaler().fit_transform(X_train)

    dt = datetime.now()#计时用的
    print('time is ' + dt.strftime('%Y-%m-%d %H:%M:%S'))
    
    #调用函数进行训练，参数可以自己调，效果可能更好
    model_svc = svm.SVC(kernel='rbf', gamma='scale')
    model_svc.fit(x, y)

    dt = datetime.now()
    print('time is ' + dt.strftime('%Y-%m-%d %H:%M:%S'))

    return model_svc

def test(model_svc):
    x, y = Getdata(path[2],path[3])  # 加载测试集
    #x = preprocessing.StandardScaler().fit_transform(test_images)
    print(model_svc.score(x, y))  
    y_pred = model_svc.predict(x)
    print("准确率： %.6f" % sm.accuracy_score(y,y_pred))
    print("F1-score: %.6f" % sm.f1_score(y,y_pred,average='macro'))
    print("召回率： %.6f" % sm.recall_score(y,y_pred,average='macro'))
    #return model_svc.score(x_test, y_test)
    return x, y

def pred(model_svc, pred_num, test_images, test_labels):
    y_pred = model_svc.predict(test_images[9690 - pred_num:9690])  # 进行预测,能得到一个结果
    print(y_pred)
 
    X_show = test_images[9690 - pred_num:9690]
    #Y_show = test_labels[9690 - pred_num:9690]
 
    #打印图片看看效果
    for i in range(pred_num):
        x_show = X_show[i].reshape(28, 28)
        plt.subplot(1, pred_num, i + 1)
        plt.imshow(x_show, cmap=plt.cm.gray_r)
        plt.title(str(y_pred[i]))
        plt.axis('off')
    plt.show()

model = train()#训练个数
test_images, test_labels = test(model)
pred(model,9,test_images, test_labels)