# -*- coding: utf-8 -*-
import numpy as np
import cv2
import mtcnn_face_detector as fd
import detector_predict as dp
# capture = cv2.VideoCapture(0)
# while (True):
#     ref, frame = capture.read()
#     boundingboxes, points = fd.detector_face(frame)
#     if (boundingboxes.size == 0):
#         print('No One!')
#         cv2.imshow('1', frame)
#     else:
#     	out = dp.face_align_predict(frame,points,boundingboxes)
#     	print(out)
#     cv2.waitKey(1)
path = 'test_0002.jpg'
img = cv2.imread(path)
out = dp.predict(img)
print(out)
