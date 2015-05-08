import os
import matplotlib.pyplot as plt
import cv2

#print os.getcwd()
#os.chdir('/home/feixuan/caffe-master')
#print os.getcwd()

caffe_root='/home/feixuan/caffe-master/'

import sys
sys.path.insert(0,caffe_root+ "distribute/python")

import caffe

caffe.set_phase_test()

net=caffe.Classifier('/home/feixuan/caffe-master/data/fp/deproy.prototxt','/home/feixuan/caffe-master/data/fp/temp_iter_10000.caffemodel')
net.set_phase_test()
net.set_mode_gpu()

img=caffe.io.load_image('/home/feixuan/caffe-master/data/fp/test/1.jpg',False)
print img.shape
prediction = net.predict([img])
print 'prediction shape:', prediction[0].shape
points=prediction[0]
recoverfacepoint=points
s=39;
recoverfacepoint[0::2]=[int(j*39.0/2+39.0/2) for j in points[0::2]]
recoverfacepoint[1::2]=[int(j*39.0/2+39.0/2) for j in points[1::2]]
face=caffe.io.load_image('9.jpg',False)

cv2.circle(face,(recoverfacepoint[0],recoverfacepoint[1]),1,(255,0,0),1)
cv2.imshow('face',face)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.circle(face,(recoverfacepoint[2],recoverfacepoint[3]),1,(255,0,0),1)
cv2.imshow('face',face)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.circle(face,(recoverfacepoint[4],recoverfacepoint[5]),1,(255,0,0),1)
cv2.imshow('face',face)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.circle(face,(recoverfacepoint[6],recoverfacepoint[7]),1,(255,0,0),1)
cv2.circle(face,(recoverfacepoint[8],recoverfacepoint[9]),1,(255,0,0),1)
                        
cv2.imshow('face',face)
cv2.waitKey(0)
cv2.destroyAllWindows()
