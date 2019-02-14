# -*- coding: utf-8 -*-
import os
import numpy as np
caffe_root = '/home/app/caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

def load_model():
	model_deploy = './net/deploy.prototxt'
	model_weight = './model/solver_iter_31000.caffemodel'

	net = caffe.Net(model_deploy,      # defines the structure of the model
                model_weight,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)
	mu = np.load('./model/emotion_mean.npy')
	# create transformer for the input called 'data'
	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

	transformer.set_transpose('data', (2,0,1))     # move image channels to outermost dimension
	transformer.set_mean('data', mu)               # subtract the dataset-mean value in each channel
	# transformer.set_raw_scale('data', 255)         # rescale from [0, 1] to [0, 255]
	# transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

	net.blobs['data'].reshape(1,         # batch size
	                          3,         # 3-channel (BGR) images
	                          224, 224)  # image size is 227x227
	return net,transformer
