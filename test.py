import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
import time
caffe_root='/home/feixuan/caffe-master/'

import sys
sys.path.insert(0,caffe_root+"distribute/python")

import caffe

net=caffe.Classifier('/home/feixuan/caffe-master/data/fp/deproy.prototxt','/home/feixuan/caffe-master/data/fp/temp/_iter_100000.caffemodel')


#define the variance
height=39
width=39
labeldim=10

dirname='/home/feixuan/train/'

#filename='/home/feixuan/train/trainImageList.txt'
filename='/home/feixuan/train/testImageList.txt'

f=open(filename)
line=f.readline()

#redirection the print
temp_print=sys.stdout
sys.stdout=open('/home/feixuan/caffe-master/record.txt','w')

i=0
total_loss=0
list_loss=[]
#record the time before procesing
time1=time.time()

while line and i<100:
	print '\033[1;31;40m'
	print '*'*50
	print 'Processing:',i,'\n'
	content=line.split(' ')
	content[1:]=[int(math.floor(float(j))) for j in content[1:]]
	imgname=dirname+content[0].replace("\\",'/')
	print imgname
	

#get the real position of facial point
	facewidth=content[4]-content[3]+1
	faceheight=content[2]-content[1]+1
	center_x=content[1]+math.floor(facewidth/2)
	center_y=content[3]+math.floor(faceheight/2)

	realpoint=content[5:]
	realpoint[0::2]=[float(j-center_x)/(float(facewidth)/2) for j in realpoint[0::2]]
	realpoint[1::2]=[float(j-center_y)/(float(faceheight)/2) for j in realpoint[1::2]]

	for j in realpoint:
		assert(1>=j>=0 or 0>=j>=-1)
    
	

#predict the facial point of the input image
	img=caffe.io.load_image(imgname,False)
	img=img[:,:,0]
	img=np.reshape(img,(250,250,1))

	prediction = net.predict([img])
#	print 'prediction shape:',prediction[0].shape
	points=prediction[0]

	print '\nprediction point:'
	for index in range(len(points)):
		if(index%2==0):
			print points[index],points[index]
	print '\nreal point:'
	for index in range(len(realpoint)):
		if(index%2==0):
			print '%.6f %.6f' %(realpoint[index],realpoint[index+1])

#calculate the sum of  square loss
	sum_loss=np.sqrt(sum([pow(a-b,2) for a,b in zip(realpoint,points)]))
	total_loss+=sum_loss
	list_loss.append(sum_loss)

	print '\nprediction loss:',sum_loss
	print '*'*50

#annotate the predicted point on the face 
#	recoverfacepoint=points
#	recoverfacepoint[0::2]=[int((j*39.0/2+39.0/2)*facewidth/39.0+content[1]) for j in points[0::2]]
#	recoverfacepoint[1::2]=[int((j*39.0/2+39.0/2)*faceheight/39.0+content[3]) for j in points[1::2]]
#
#	face=caffe.io.load_image(imgname,False)
#	cv2.circle(face,(recoverfacepoint[0],recoverfacepoint[1]),1,(255,255,0),1)
#	cv2.circle(face,(recoverfacepoint[2],recoverfacepoint[3]),1,(255,255,0),1)
#	cv2.circle(face,(recoverfacepoint[4],recoverfacepoint[5]),1,(255,255,0),1)
#	cv2.circle(face,(recoverfacepoint[6],recoverfacepoint[7]),1,(255,255,0),1)
#	cv2.circle(face,(recoverfacepoint[8],recoverfacepoint[9]),1,(255,255,0),1)
#                        
#	cv2.imshow('face',face)
##	cv2.waitKey(0)
#	cv2.destroyAllWindows()

	line=f.readline()
	i+=1
#	print i,line
time2=time.time()
time_spend=(time2-time1)

#recover the stdout
sys.stdout.close()
sys.stdout=temp_print

#output the total loss
print '\033[1;31;40m'
print 'tested images:',i
print 'time spend: %.4f'%time_spend,'s','\n'
print 'total prediction loss:',total_loss
print 'average prediction loss:',total_loss/i
print '\033[0m'

#plot the individual loss
plt.plot(list_loss,color='blue',linewidth=2)
plt.xlabel('test images')
plt.ylabel('predict loss')
plt.ylim(0,1)
plt.title('face point detection')
plt.legend()
plt.show()

