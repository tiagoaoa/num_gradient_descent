# -*- coding: utf-8 -*-
#Functions for the implementation of Adversarial Attacks on Machine Learning Image Classifiers.
# Tiago Alves <tiago@ime.uerj.br>

import numpy as np
from numdifftools import Gradient as ndGradient
import random
import scipy.stats as stats

exp_lambda = 30.0

def gray_to_rgb(height, width, image_gray):
		image_rgb = np.zeros((height, width,3))
		#image_rgb = np.array([])
		for row in range(0,height):
			for col in range(0, width):
				for c in range(3):
					image_rgb[row][col][c] = image_gray[row*width+col]
		
		return image_rgb.astype('uint8')



def f_noise(f,x):
		#Adds short tailed distributed noise to function result
		u = random.uniform(0,1)
		#u = random.randint(2,10)
		noise = stats.expon.rvs(scale=1/exp_lambda) #exponential
		#noise = np.exp(-1*u) #exponential
		noise = u > 0.5 and noise or -1 * noise
		dist = f(x) + noise
		##print "Noise %f"%noise
		return dist 


def grad_ascent(f, x):
		#print x
		conf = f(x)
		while conf < 0.8:
			grad = nd.Gradient(f)(x)
			x += grad
			conf = f(x)
			#count += 1

		return x





def num_grad(f, x):
		delta = 1
		grad = np.zeros(len(x))
		a = np.copy(x)
		#print len(x)
		for i in range(len(x)):
			a[i] = x[i] + delta
			grad[i] = (f(a) - f(x))/delta
			a[i] -= delta
	
		return grad

def num_ascent(f, x):
	conf =  f(x)
	print("Conf is {}".format(conf))
	count = 0
	while conf < 0.4:
		grad = num_grad(f,x)
		#grad = ndGradient(f)(x)
		print(grad)
		x += grad

		conf = f(x)
		print("Conf {}".format(conf))




	
	return x




def check_classification(f, img, label):
	image_rgb = np.reshape(img, (3, 32, 32))
	image_rgb = image_rgb.transpose(1,2,0)					
	#from PIL import Image
	#print image_rgb.shape
	#img = Image.fromarray(image_rgb, 'RGB')
	#img.show()
	
	image_rgb = transform_fn(nd.array(image_rgb))
	#print "%s %s"%(label, f(image_rgb))
	return label == f(image_rgb)


def progressbar(n, total):
	p = int(float(n)/total*100)
	#print "%d %d" %(p, n)
	bar = '█'*p + ' '*(100-p)
	sys.stderr.write('\r[%s] %s%s ...\r' % (bar, p, '%'))	
	if n == total:
		sys.stderr.write('\n')

def test_accuracy(f):
	import cPickle as pickle
	fp = open("cifar-10-batches-py/data_batch_%d" %batch_number, 'rb')
	batch_dict = pickle.load(fp)
	count = 0
	i = 1
	k = len(batch_dict["labels"])
	for img, label in map(None, batch_dict["data"], batch_dict["labels"]):
		if check_classification(f, img, class_names[label]):
			count += 1
		if count % 10 == 0:
			#print "%d %d" %(count, len(batch_dict["data"]))
			progressbar(i, k)
		i += 1
	return float(count)/len(batch_dict["data"])

def rgb_to_gray(pixels):
	#img_array = 0.2125 * pixels[:,:,0] + 0.7154 * pixels[:,:,1] + 0.0721 * pixels[:,:,2]
	r, g, b = pixels[:,:,0] , pixels[:,:,1] , pixels[:,:,2]
	img_array = 0.2125 * r + 0.7154 * g + 0.0721 * b

	height, width = img_array.shape
	print("Shape {}".format(img_array.shape))	
	return (height, width, img_array.reshape(width*height).astype(int))

