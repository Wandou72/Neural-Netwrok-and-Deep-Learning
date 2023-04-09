# 所有使用的包
import numpy as np
import struct
import matplotlib.pyplot as plt
import math
import pickle


# 定义解析文件读取图像数据的函数
def images_data(dtype):
    if dtype == 'train':
        data = open('./MNIST/train-images-idx3-ubyte', 'rb').read()
        
    else:
        data = open('./MNIST/t10k-images-idx3-ubyte', 'rb').read()
    index = 0
    # 文件头信息：魔数、图片数、每张图高、图宽
    # 32位整型采用I格式
    fmt_header = '>IIII'
    magicnum, imagenum, rownum, colnum = struct.unpack_from(fmt_header, data, index)
    # 数据在缓存中的指针位置 index此时为16
    index += struct.calcsize('>IIII')  

    output = np.empty((imagenum, rownum * colnum))
    # 图像数据像素值类型Unsigned char型(B)同时大小为28*28 784
    fmt_image = '>' + str(rownum * colnum) + 'B'
    for i in range(imagenum):
        output[i] = np.array(struct.unpack_from(fmt_image, data, index)).reshape((1, rownum * colnum))/255.
        index += struct.calcsize(fmt_image)
    mu = np.mean(output, axis=1, keepdims=True)
    sigma = np.std(output, axis=1, keepdims=True)
    
    data = (output - mu)/sigma
    if dtype == 'train':
        # extract 6000 pictures from training set as validation set
        ind = [i for i in range(data.shape[0])]
        np.random.seed(1)
        np.random.shuffle(ind)
        validation_data = data[ind[0:6000], :]
        train_data = data[ind[6000:], :]
        return train_data, validation_data
        
    else:
        return data



# 定义解析文件读取标签数据的函数
def labels_data(dtype):
    if dtype == 'train':
        data = open('./MNIST/train-labels-idx1-ubyte', 'rb').read()
    else:
        data = open('./MNIST/t10k-labels-idx1-ubyte', 'rb').read()
    index = 0
    # 文件头信息：魔数、标签数也就是数据量
    fmt_header = '>II'
    magicnum, labelnum = struct.unpack_from(fmt_header, data, index)

    index += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty((labelnum, 1))
    for i in range(labelnum):
        labels[i] = np.array(struct.unpack_from(fmt_image, data, index)[0]).reshape((1, 1))
        index += struct.calcsize(fmt_image)
    
    if dtype == 'train':
        # extract 6000 pictures from training set as validation set
        ind = [i for i in range(labels.shape[0])]
        np.random.seed(1)
        np.random.shuffle(ind)
        validation_labels = labels[ind[0:6000], :]
        train_labels = labels[ind[6000:], :]
        return train_labels, validation_labels
        
    else:
        return labels


# 定义函数实现对数据集调用训练好模型的功能
def predictbymodel(testdata, testlabel):
    paras = pickle.load(open('bestpara.pkl', 'rb'))
    hiddenlayer_output = np.maximum(np.matmul(testdata, paras['W1']) +  paras['b1'], 0)
    outlayer = np.maximum(np.matmul(hiddenlayer_output,  paras['W2']) +  paras['b2'], 0)
    prediction = np.argmax(outlayer, axis=1).reshape((len(testdata),1))
    accuracy = np.mean(prediction == testlabel)
    return accuracy

if __name__ == '__main__':
    images_data('train')