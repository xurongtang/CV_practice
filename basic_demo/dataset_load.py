import torch
import cv2,os
import numpy as np
import struct
import matplotlib.pyplot as plt

def __decode_idx3_ubyte(file):
    """
    解析数据文件
    """
    # 读取二进制数据
    with open(file, 'rb') as fp:
        bin_data = fp.read()
    
    # 解析文件中的头信息
    # 从文件头部依次读取四个32位，分别为：
    # magic，numImgs, numRows, numCols
    # 偏置
    offset = 0
    # 读取格式: 大端
    fmt_header = '>iiii'
    magic, numImgs, numRows, numCols = struct.unpack_from(fmt_header, bin_data, offset)
    # print(magic,numImgs,numRows,numCols)
    
    # 解析图片数据
    # 偏置掉头文件信息
    offset = struct.calcsize(fmt_header)
    # 读取格式
    fmt_image = '>'+str(numImgs*numRows*numCols)+'B'
    data = torch.tensor(struct.unpack_from(fmt_image, bin_data, offset)).reshape(numImgs, numRows, numCols)
    return data

def __decode_idx1_ubyte(file):
    """
    解析标签文件
    """
    # 读取二进制数据
    with open(file, 'rb') as fp:
        bin_data = fp.read()
    
    # 解析文件中的头信息
    # 从文件头部依次读取两个个32位，分别为：
    # magic，numImgs
    # 偏置
    offset = 0
    # 读取格式: 大端
    fmt_header = '>ii'
    magic, numImgs = struct.unpack_from(fmt_header, bin_data, offset)
    # print(magic,numImgs)
    # 解析图片数据
    # 偏置掉头文件信息
    offset = struct.calcsize(fmt_header)
    # 读取格式
    fmt_image = '>'+str(numImgs)+'B'
    data = torch.tensor(struct.unpack_from(fmt_image, bin_data, offset))
    return data

def get_MINIST():
    # 文件路径
    data_path = 'E:/ComputerVision_Proj/dataset/MINIST/DATA/'
    file_names = ['t10k-images-idx3-ubyte',
                    't10k-labels-idx1-ubyte',
                    'train-images-idx3-ubyte',
                    'train-labels-idx1-ubyte']
    test_set = (__decode_idx3_ubyte(os.path.join(data_path, file_names[0])),
                __decode_idx1_ubyte(os.path.join(data_path, file_names[1])))
    train_set = (__decode_idx3_ubyte(os.path.join(data_path, file_names[2])),
                __decode_idx1_ubyte(os.path.join(data_path, file_names[3])))
    return train_set, test_set

def get_CIFAR10():
    root_path = 'E:/ComputerVision_Proj/dataset/cifar-10-batches-py/'
    train_path = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    import pickle
    res_data = []
    res_labels = []
    for i in range(5):
        with open(os.path.join(root_path, root_path+train_path[i]), 'rb') as fp:
            res = pickle.load(fp, encoding='bytes')
        labels = res[b'labels']
        data = res[b'data']
        assert len(labels) == len(data)
        dataset_input = np.zeros((len(labels), 3, 32, 32))
        dataset_labels = np.array(labels,dtype=np.uint8)
        for j in range(len(labels)):
            dataset_input[j] = data[j].reshape(3, 32, 32)
        res_data.append(dataset_input)
        res_labels.append(dataset_labels)
    all_train_dataset_input = np.concatenate(res_data, axis=0)
    all_train_dataset_labels = np.concatenate(res_labels, axis=0)
    # test dataset
    with open(os.path.join(root_path, 'test_batch'), 'rb') as fp:
        res = pickle.load(fp, encoding='bytes')
    labels = res[b'labels']
    data = res[b'data']
    assert len(labels) == len(data)
    test_input = np.zeros((len(labels), 3, 32, 32))
    test_labels = np.array(labels,dtype=np.uint8)
    for j in range(len(labels)):
        test_input[j] = data[j].reshape(3, 32, 32)
    # 转化
    train_input = torch.tensor(all_train_dataset_input)
    train_labels = torch.tensor(all_train_dataset_labels)
    test_input = torch.tensor(test_input)
    test_labels = torch.tensor(test_labels)
    return (train_input,train_labels),(test_input,test_labels)

if __name__ == '__main__':
    
    # train_set, test_set = get_MINIST()
    # print(train_set[0].shape,train_set[1].shape)
    # print(test_set[0].shape,test_set[1].shape)
    # test = np.array(train_set[0][0])
    # plt.imshow(test)
    # plt.show()
    
    train_set,test_set = get_CIFAR10()
    print(train_set[0].shape,train_set[1].shape)
    print(test_set[0].shape,test_set[1].shape)
