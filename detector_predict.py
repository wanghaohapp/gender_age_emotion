# -*- coding: utf-8 -*-
import numpy as np
caffe_root = '/home/app/caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import load_model as lm
import cv2
caffe.set_mode_gpu()
caffe.set_device(1)
net,transformer = lm.load_model()
def face_align_predict(frame,points,boundingboxes):
	x1 = boundingboxes[:, 0]
	y1 = boundingboxes[:, 1]
	x2 = boundingboxes[:, 2]
	y2 = boundingboxes[:, 3]
	out = []
	for i in range(x1.shape[0]):
		tem_out = []
		img = frame.copy()
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
		for j in range(data1.shape[0]):
			tem_out.append(float('%.6f' % data1[j]))
		tem_out.append(data2.argmax())
		tem_out.append(data3.argmax())
		tem_out.append(float('%.2f' % data4[0]))
		out.append(tem_out)
	return out
def predict(img):
	transformed_image = transformer.preprocess('data', img)
	net.blobs['data'].data[...] = transformed_image
	output = net.forward()
	data1 = net.blobs['prob_emotion'].data[0]
	data2 = net.blobs['prob_gender'].data[0]
	data3 = net.blobs['prob_age'].data[0]
	data4 = net.blobs['prob_three_classify'].data[0]
	out = []
	for j in range(data1.shape[0]):
		out.append(float('%.6f' % data1[j]))
	out.append(data2.argmax())
	out.append(data3.argmax())
	out.append(float('%.2f' % data4[0]))
	return out
			
