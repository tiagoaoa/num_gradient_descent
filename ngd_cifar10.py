'''CIFAR10 Security POC
Tiago Alves <tiago@ime.uerj.br>'''

'''This portion of the code is based on the example provided Liu Kuan (https://github.com/kuangliu/pytorch-cifar). 
However, the attacks/defenses in ngd_attacks.py can be used in any other implementations of CIFAR10 classifiers with almost no adaption required.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import numpy as np

import os
import argparse

from models import *
from utils import progress_bar

from PIL import Image

import ngd_attacks as ngd

width, height = (32, 32)


parser = argparse.ArgumentParser(description='CIFAR10 Security Attacks')
parser.add_argument('--input-pic', '-i', type=str, help='Input image', required=False)
parser.add_argument('--target', type=str, help='Target class', required=False)
args = parser.parse_args()



transform_test = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

transform_fn = transforms.Compose([
	transforms.Resize(32),
	transforms.CenterCrop(32),
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
#])

# Model
#print('==> Building model..')
net = VGG('VGG19')
#net = ResNet18()
# net = PreActResNet18()
#net = GoogLeNet()
#net = DenseNet121()
# net = ResNeXt29_2x64d()
#net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
#net = ShuffleNetV2(1)
#net = net.to(device)
#if device == 'cuda':
net = torch.nn.DataParallel(net)
#	cudnn.benchmark = True

	# Load checkpoint.
print('==> Resuming from checkpoint..')
#checkpoint = torch.load('./checkpoint_lenet/ckpt.t8', map_location=torch.device('cpu'))
#checkpoint = torch.load('./checkpoint_resnet/ckpt.t8', map_location=torch.device('cpu'))
checkpoint = torch.load('./checkpoint/ckpt_vgg19.t9')#, map_location=torch.device('cpu'))
#checkpoint = torch.load('./checkpoint/ckpt_vggblack.t9', map_location=torch.device('cpu'))

#print(checkpoint)
net.load_state_dict(checkpoint['net'])
#est_acc = checkpoint['acc']
print('Resumed')
#tart_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


def test(f=net):
	net.eval()
	test_loss = 0
	correct = 0
	total = 0
	with torch.no_grad():
		for batch_idx, (inputs, targets) in enumerate(testloader):
			inputs, targets = inputs.to(device), targets.to(device)
			outputs = f(inputs)
			loss = criterion(outputs, targets)

			test_loss += loss.item()
			_, predicted = outputs.max(1)
			#print("{} -- {}".format(targets, predicted))
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()

			progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
			% (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


def save_img(img, count=None):
	if count != None:
		img = transforms.ToPILImage()(img)
		img.save('output{}.jpg'.format(count))






def test_classifier(h, w, x):
	#x *= 255
	pixels = x.reshape((h, w, 3)).astype('uint8')
	
	img = Image.fromarray(pixels, mode='RGB')
	img = transform_fn(img)
	print(img)
	output = net(img.unsqueeze(dim=0))
	output = F.softmax(output[0], dim=0)

	print(output)
	save_img(img, count=0)

	value, index = torch.max(output, 0)
	print("{} -- {}".format(value, classes[index]))


def save_transform(h, w, x, save_img=None):
	#x *= 255	
	img = x.reshape((h, w, 3)).astype('uint8')
	img = Image.fromarray(img, mode='RGB')
	img.save('output.jpg')
	if save_img != None:
		img.save('imgs/output{}.jpg'.format(save_img))
	img = Image.open('output.jpg')
	img = transform_fn(img)
	return img

def create_f(h, w, target):
	def f(x, save_img=None):
		pixels = save_transform(h, w, x, save_img)
		output = net(pixels.unsqueeze(dim=0))
		output = F.softmax(output[0], dim=0)
		return output[target].item()
	#return lambda x: f(x, target)
	return f
	



def linearize_pixels(img):
	x = np.copy(np.asarray(img))
	h, w, c = x.shape
	img_array = x.reshape(h*w*c).astype('float64')
	#img_array /= 255
	return h, w, img_array
	
	








	






if args.input_pic:
	net.eval()
	print("There is input pic")
	img = Image.open(args.input_pic)
	h, w, img_array = linearize_pixels(img)	


	#ith torch.autograd.profiler.profile(use_cuda=True) as prof:
	test_classifier(h, w, img_array)
	#rint(prof)
	
	if args.target:
		f = create_f(h, w, classes.index(args.target))

		print(f(img_array))

		ngd.num_ascent(f, img_array)
		


else:
	print("No input pic provided.")
        #You can call test(f=function_to_be_called_for_predictions) to test the accuracy, the default is f=net.

	
	


