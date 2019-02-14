# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np
import random
import os
import h5py
img_root = '/home/app/python/age_gender_image/fer2013'
train_path = '/home/app/python/age_gender_model/age_gender/face_feature/label/emotion/hdf5/train3.txt'
test_path = '/home/app/python/age_gender_model/age_gender/face_feature/label/emotion/hdf5/val.txt'
train_out = '/home/app/python/age_gender_model/age_gender/face_feature/data/emotion/hdf5/vgg/hdf5_train'
test_out = '/home/app/python/age_gender_model/age_gender/face_feature/data/emotion/hdf5/vgg/hdf5_test.h5'
image_dir = [train_path,test_path]
hdf5_file = [train_out,test_out]

with open(train_path) as f:
    lines = f.readlines()

file_list = []
labels = np.zeros((len(lines), 2)).astype(np.int)
datas = np.zeros((len(lines),3,224,224)).astype(np.float32)
count = 0
for line in lines:
    file_list.append(line.split()[0])
    labels[count][0] = line.split()[1]
    labels[count][1] = line.split()[2]
    count += 1
f.close()

for ii, _file in enumerate(file_list):
    path = os.path.join(img_root,_file)
    image = cv.imread(path)
    image = cv.resize(image,(224,224))
    img = np.array(image)
    img = img.transpose(2,0,1)
    datas[ii, :, :, :] = img.astype(np.float32)
mean = datas.mean(axis=0)
mean = mean.mean(1).mean(1)
for i in range(len(datas)):
    datas[i][0] = datas[i][0] - mean[0]
    datas[i][1] = datas[i][1] - mean[1]
    datas[i][2] = datas[i][2] - mean[2]

#分离训练数据
long_datas = len(datas) / 5
datas_0 = datas[ : long_datas,:,:,:]
datas_1 = datas[long_datas : (2*long_datas),:,:,:]
datas_2 = datas[(2*long_datas) : (3*long_datas),:,:,:]
datas_3 = datas[(3*long_datas) : (4*long_datas),:,:,:]
datas_4 = datas[(4*long_datas) : ,:,:,:]
labels_0 = labels[ : long_datas,:]
labels_1 = labels[long_datas : (2*long_datas),:]
labels_2 = labels[(2*long_datas) : (3*long_datas),:]
labels_3 = labels[(3*long_datas) : (4*long_datas),:]
labels_4 = labels[(4*long_datas) : ,:]
print(len(datas_0))
print(len(datas_1))
print(len(datas_2))
print(len(datas_3))
file_path = []
for i in range(5):
    file_path.append(train_out + '{0}.h5'.format((i+10)))
with h5py.File(file_path[0],'w') as fout:
    fout.create_dataset('data',data = datas_0)
    fout.create_dataset('label', data=labels_0)
fout.close()
with h5py.File(file_path[1],'w') as fout:
    fout.create_dataset('data',data = datas_1)
    fout.create_dataset('label', data=labels_1)
fout.close()
with h5py.File(file_path[2],'w') as fout:
    fout.create_dataset('data',data = datas_2)
    fout.create_dataset('label', data=labels_2)
fout.close()
with h5py.File(file_path[3],'w') as fout:
    fout.create_dataset('data',data = datas_3)
    fout.create_dataset('label', data=labels_3)
fout.close()
with h5py.File(file_path[4],'w') as fout:
    fout.create_dataset('data',data = datas_4)
    fout.create_dataset('label', data=labels_4)
fout.close()

t = open('/home/app/python/age_gender_model/age_gender/face_feature/data/emotion/hdf5/vgg/train_list.txt','w+')
for yy in file_path:
    t.write(yy)
    t.write('\n')
t.close()
# #读取test data
# with open(test_path) as f1:
#     lines1 = f1.readlines()
# file_list = []
# test_labels = np.zeros((len(lines1), 2)).astype(np.int)
# test_datas = np.zeros((len(lines1),3,224,224)).astype(np.float32)
# count = 0
# for line1 in lines1:
#     file_list.append(line1.split()[0])
#     test_labels[count][0] = line1.split()[1]
#     test_labels[count][1] = line1.split()[2]
#     count += 1
# f1.close()
#
# for ii, _file in enumerate(file_list):
#     path = os.path.join(img_root,_file)
#     image = cv.imread(path)
#     image = cv.resize(image,(224,224))
#     img = np.array(image)
#     img = img.transpose(2,0,1)
#     test_datas[ii, :, :, :] = img.astype(np.float32)
# for i in range(len(test_datas)):
#     test_datas[i][0] = test_datas[i][0] - mean[0]
#     test_datas[i][1] = test_datas[i][1] - mean[1]
#     test_datas[i][2] = test_datas[i][2] - mean[2]
#
# with h5py.File(test_out,'w') as fout1:
#     fout1.create_dataset('data', data = test_datas)
#     fout1.create_dataset('label', data = test_labels)
# fout1.close()
#np.save('/home/app/python/age_gender_model/age_gender/face_feature/model/hdf5/mean.npy', mean)

