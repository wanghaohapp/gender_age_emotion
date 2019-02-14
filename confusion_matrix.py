# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2 as cv

caffe_root = '/home/app/caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import mtcnn_face_detector as fd
import time
caffe.set_mode_gpu()

#读入网络模型
model_deploy = '/home/app/program/emotion_gender_age/net/deploy.prototxt'
model_weight = '/home/app/program/emotion_gender_age/model/age/solver_iter_28000.caffemodel'

net = caffe.Net(model_deploy,      # defines the structure of the model
                model_weight,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)
mu = np.load('/home/app/program/emotion_gender_age/model/beiyou_emotion/emotion_mean.npy')
# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))     # move image channels to outermost dimension
transformer.set_mean('data', mu)               # subtract the dataset-mean value in each channel
# transformer.set_raw_scale('data', 255)         # rescale from [0, 1] to [0, 255]
# transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

net.blobs['data'].reshape(1,         # batch size
                          3,         # 3-channel (BGR) images
                          224, 224)  # image size is 227x227


def predict(img):
    transformed_image = transformer.preprocess('data', img)
    net.blobs['data'].data[...] = transformed_image
    output = net.forward()
    data1 = net.blobs['prob_emotion'].data[0]
    return data1.argmax()

def load_data(root_img_path):
    data_path = '/home/app/program/emotion_gender_age/emotion_label.txt'
    f_read = open(data_path)
    datas = f_read.readlines()
    f_read.close()
    return datas

root_img_path = '/home/app/data/beiyou/basic/Image/face_aligned'
#获取数据与标签
datas = load_data(root_img_path)
#定义标签类别与混淆矩阵
dim = 7
confusion_matrix = np.zeros((dim,dim),np.int)
#获取混淆矩阵
for data in datas:
    image_path = os.path.join(root_img_path,data.split()[0])
    label = data.split()[1]

    #获取混淆矩阵
    img = cv.imread(image_path)
    predict_value = int(predict(img))
    confusion_matrix[label,predict_value] += 1
print(confusion_matrix)

