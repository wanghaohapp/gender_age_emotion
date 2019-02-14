# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import dlib
caffe_root = '/home/app/caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import mtcnn_face_detector as fd
import time
caffe.set_mode_gpu()
caffe.set_device(1)
# plt.rcParams['figure.figsize'] = (10, 10)
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'

#导入平均模型
# mean_filename='/home/app/python/age_gender_model/age_gender/face_feature/model/gender/gender_mean.binaryproto'
# #mean_filename='/home/app/python/age_gender_model/age_gender/faces1/model/gender_mean.binaryproto'
# proto_data = open(mean_filename, "rb").read()
# a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
# mean  = caffe.io.blobproto_to_array(a)[0]
# mu = mean.mean(1).mean(1)
#导入年纪模型
# age_net_pretrained='/home/app/python/model/AgeGenderDeepLearning-master/models/age_net.caffemodel'
# age_net_model_file='/home/app/python/model/AgeGenderDeepLearning-master/age_net_definitions/deploy.prototxt'
# age_net_pretrained='/home/app/python/age_gender_model/age_gender/faces1/model/age.caffemodel'
# age_net_model_file='/home/app/python/age_gender_model/age_gender/AgeGenderDeepLearning-master/age_net_definitions/deploy.prototxt'
# age_net = caffe.Classifier(age_net_model_file, age_net_pretrained,
#                        mean=mean,
#                        channel_swap=(2,1,0),
#                        raw_scale=255,
#                        image_dims=(256, 256))

#导入性别模型
# gender_net_pretrained='/home/app/python/age_gender_model/age_gender/face_feature/model/gender/solver_gender_iter_30000.caffemodel'
# gender_net_model_file='/home/app/python/age_gender_model/age_gender/face_feature/net/gender/deploy.prototxt'
# # gender_net_pretrained='/home/app/python/age_gender_model/age_gender/faces1/model/age_gender.caffemodel'
# gender_net_model_file='/home/app/python/age_gender_model/age_gender/faces1/net/hdf5/deploy.prototxt'
# gender_net = caffe.Classifier(gender_net_model_file, gender_net_pretrained,
#                        mean=mean,
#                        channel_swap=(2,1,0),
#                        raw_scale=255,
#                        image_dims=(256, 256))

#gender_net = caffe.Classifier(gender_net_model_file, gender_net_pretrained,mean=mean,raw_scale=255,image_dims=(256, 256))
# #label
model_deploy = '/home/app/program/emotion_gender_age/net/deploy.prototxt'
model_weight = '/home/app/program/emotion_gender_age/model/three_classify/solver_iter_31000.caffemodel'

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

# consume_time = '/home/app/python/age_gender_model/age_gender/faces1/time/age_gender_gpu.txt'
# f = open(consume_time, 'w')
emotion_list = ['Surprise','Fear','Disgust','Happy','Sadness','Anger','Neutral']
age_list=['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
gender_list=['Male','Female']
#cnn_face_detector = dlib.cnn_face_detection_model_v1('/home/app/python/mmod_human_face_detector.dat')
#detector = dlib.get_frontal_face_detector()
# capture = cv2.VideoCapture(0)
# while (True):
#     ref, frame = capture.read()
#     #input_image = caffe.io.load_image(frame)
#     #dets = detector(frame, 1)
#     #face_detector_start = time.clock()
#     #start = time.clock()
#     #dets = cnn_face_detector(frame, 1)
#     #face_detector_end = time.clock()
#     #s = '人脸检测耗时： {0}s'.format(face_detector_end - face_detector_start)
#     #f.write(s + '\n')
#     #print(s)
#     # if len(dets):
#     #     for i, d in enumerate(dets):
#     #         #img = frame[d.top():d.bottom(), d.left():d.right()]
#     #          #pre_image_start = time.clock()
#     #         if ((d.rect.top() >= 0) & (d.rect.bottom() >= 0) & (d.rect.left() >= 0) & (d.rect.right() >= 0)):
#     #             img = frame[d.rect.top():d.rect.bottom(), d.rect.left():d.rect.right()]
#     #             transformed_image = transformer.preprocess('data', img)
#     #             # pre_image_end = time.clock()
#     #             # s1 = '人脸影像预处理耗时： {0}s'.format(pre_image_end - pre_image_start)
#     #             # f.write(s1 + '\n')
#     #             # print(s1)
#     #             #net_predict_start = time.clock()
#     #             net.blobs['data'].data[...] = transformed_image
#     #             output = net.forward()
#     #             data1 = net.blobs['prob_emotion'].data[0]
#     #             data2 = net.blobs['prob_gender'].data[0]
#     #             data3 = net.blobs['prob_age'].data[0]
#     #             #prediction = age_net.predict([img])
#     #             # net_predict_end = time.clock()
#     #             # s2 = '人脸性别年龄识别耗时： {0}s'.format(net_predict_end - net_predict_start)
#     #             # f.write(s2 + '\n')
#     #             # print(s2)
#     #             print('{0}_predicted age:'.format(i), emotion_list[data1.argmax()])
#     #
#     #             #prediction = gender_net.predict([img])
#     #
#     #             print('{0}_predicted gender:'.format(i), gender_list[data2.argmax()])
#     #             print('{0}_predicted gender:'.format(i), age_list[data3.argmax()])
#     #             #print(prediction)
#     #             #print('{0}_predicted gender:'.format(i), gender_list[prediction[0].argmax()])
#     #             # cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 2)
#     #             cv2.rectangle(frame, (d.rect.left(), d.rect.top()), (d.rect.right(), d.rect.bottom()), (0, 255, 0), 2)
#     #             cv2.imshow('1', frame)
#     #         else:
#     #             print('No One!')
#     #             cv2.imshow('1', frame)
#     #
#     # else:
#     #     print('No One!')
#     #     cv2.imshow('1', frame)
#     boundingboxes, points = fd.detector_face(frame)
#     if (boundingboxes.size == 0):
#         print('No One!')
#         cv2.imshow('1', frame)
#     else:
#         x1 = boundingboxes[:, 0]
#         y1 = boundingboxes[:, 1]
#         x2 = boundingboxes[:, 2]
#         y2 = boundingboxes[:, 3]
#         for i in range(x1.shape[0]):
#             img = frame.copy()
#             point = points[i]
#             keypoint1 = np.float32([[30, 30], [70, 30], [50, 80]])
#             keypoint2 = point[:2, :]
#             x = point[3, :]
#             y = point[4, :]
#             x = (x + y) / 2.0
#             keypoint2 = np.row_stack((keypoint2, x))
#             matrix = cv2.getAffineTransform(keypoint2, keypoint1)
#             output = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))
#             face_img = output[:100, :100]
#             transformed_image = transformer.preprocess('data', face_img)
#             net.blobs['data'].data[...] = transformed_image
#             output = net.forward()
#             data1 = net.blobs['prob_emotion'].data[0]
#             data2 = net.blobs['prob_gender'].data[0]
#             data3 = net.blobs['prob_age'].data[0]
#             data4 = net.blobs['prob_three_classify'].data[0]
#             print('{0}_predicted emotion: {1}'.format(i, emotion_list[data1.argmax()]))
#             print('{0}_predicted gender: {1}'.format(i, gender_list[data2.argmax()]))
#             print('{0}_predicted age: {1}'.format(i, age_list[data3.argmax()]))
#             print('{0}_predicted happy: {1}'.format(i, float('%.2f' % data4[0])))
#             cv2.putText(frame,'emotion: {0}'.format(emotion_list[data1.argmax()]),
#                         (int(x1[i]),50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
#             cv2.putText(frame, 'gender: {0}'.format(gender_list[data2.argmax()]),
#                         (int(x1[i]), 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
#             cv2.putText(frame, 'age: {0}'.format(age_list[data3.argmax()]),
#                         (int(x1[i]), 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
#             cv2.putText(frame, 'smile_score: {0}'.format(float('%.2f' % (data4[0]*100))),
#                         (int(x1[i]), 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
#
#
#             cv2.rectangle(frame, (int(x1[i]), int(y1[i])), (int(x2[i]), int(y2[i])), (0, 255, 0), 1)
#             cv2.imshow('1', frame)
#
#     cv2.waitKey(1)


img = cv2.imread('/home/app/data/2.jpg')
img = cv2.resize(img,(500,600))
boundingboxes, points = fd.detector_face(img)
x1 = boundingboxes[:, 0]
y1 = boundingboxes[:, 1]
x2 = boundingboxes[:, 2]
y2 = boundingboxes[:, 3]
for i in range(x1.shape[0]):
    point = points[i]
    point = points[i]
    keypoint1 = np.float32([[30, 30], [70, 30], [50, 80]])
    keypoint2 = point[:2, :]
    x = point[3, :]
    y = point[4, :]
    x = (x + y) / 2.0
    keypoint2 = np.row_stack((keypoint2, x))
    matrix = cv2.getAffineTransform(keypoint2, keypoint1)
    output = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))
    face_img = output[:100, :100]
    transformed_image = transformer.preprocess('data', face_img)
    net.blobs['data'].data[...] = transformed_image
    output = net.forward()
    data1 = net.blobs['prob_emotion'].data[0]
    data2 = net.blobs['prob_gender'].data[0]
    data3 = net.blobs['prob_age'].data[0]
    data4 = net.blobs['prob_three_classify'].data[0]
    print('{0}_predicted emotion: {1}'.format(i, emotion_list[data1.argmax()]))
    print('{0}_predicted gender: {1}'.format(i, gender_list[data2.argmax()]))
    print('{0}_predicted age: {1}'.format(i, age_list[data3.argmax()]))
    print('{0}_predicted happy: {1}'.format(i, float('%.2f' % data4[0])))
    cv2.putText(img, 'emotion: {0}'.format(emotion_list[data1.argmax()]),
                (int(x1[i] - 50), 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(img, 'gender: {0}'.format(gender_list[data2.argmax()]),
                (int(x1[i] - 50), 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(img, 'age: {0}'.format(age_list[data3.argmax()]),
                (int(x1[i] - 50), 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(img, 'smile_score: {0}'.format(float('%.2f' % (data4[0] * 100))),
                (int(x1[i] - 50), 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.rectangle(img, (int(x1[i]), int(y1[i])), (int(x2[i]), int(y2[i])), (0, 255, 0), 1)
    cv2.imshow('1', img)
    cv2.imwrite('/home/app/data/2_1.jpg', img)

cv2.waitKey(1)





