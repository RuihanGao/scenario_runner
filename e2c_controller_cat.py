import os, sys
from os import path 
from glob import glob
import random
from random import shuffle

from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, ToPILImage
import torchvision.transforms.functional as F
from torchvision.transforms import ToTensor

from PIL import Image
from skimage.transform import resize
from skimage.color import rgb2gray

from tqdm import trange, tqdm
import pickle

from cat_configs import *

import numpy as np
import cv2

torch.set_default_dtype(torch.float32)

def binary_crossentropy(t, o, eps=1e-8):
	return t * torch.log(o + eps) + (1.0 - t) * torch.log(1.0 - o + eps)


def kl_bernoulli(p, q, eps=1e-8):
	# http://ufldl.stanford.edu/tutorial/unsupervised/Autoencoders/
	kl = p * torch.log((p + eps) / (q + eps)) + \
		 (1 - p) * torch.log((1 - p + eps) / (1 - q + eps))
	return kl.mean()


class NormalDistribution(object):
	"""
	Wrapper class representing a multivariate normal distribution parameterized by
	N(mu,Cov). If cov. matrix is diagonal, Cov=(sigma).^2. Otherwise,
	Cov=A*(sigma).^2*A', where A = (I+v*r^T).
	"""

	def __init__(self, mu, sigma, logsigma, *, v=None, r=None):
		self.mu = mu
		self.sigma = sigma
		self.logsigma = logsigma
		self.v = v
		self.r = r

	@property
	def cov(self):
		"""This should only be called when NormalDistribution represents one sample"""
		if self.v is not None and self.r is not None:
			assert self.v.dim() == 1
			dim = self.v.dim()
			v = self.v.unsqueeze(1)  # D * 1 vector
			rt = self.r.unsqueeze(0)  # 1 * D vector
			A = torch.eye(dim) + v.mm(rt)
			return A.mm(torch.diag(self.sigma.pow(2)).mm(A.t()))
		else:
			return torch.diag(self.sigma.pow(2))


def KLDGaussian(Q, N, eps=1e-8):
	"""KL Divergence between two Gaussians
		Assuming Q ~ N(mu0, A sigma_0A') where A = I + vr^{T}
		and      N ~ N(mu1, sigma_1)
	"""
	sum = lambda x: torch.sum(x, dim=1)
	k = float(Q.mu.size()[1])  # dimension of distribution
	mu0, v, r, mu1 = Q.mu, Q.v, Q.r, N.mu
	s02, s12 = (Q.sigma).pow(2) + eps, (N.sigma).pow(2) + eps
	a = sum(s02 * (1. + 2. * v * r) / s12) + sum(v.pow(2) / s12) * sum(r.pow(2) * s02)  # trace term
	b = sum((mu1 - mu0).pow(2) / s12)  # difference-of-means term
	c = 2. * (sum(N.logsigma - Q.logsigma) - torch.log(1. + sum(v * r) + eps))  # ratio-of-determinants term.

	#
	# print('trace: %s' % a)
	# print('mu_diff: %s' % b)
	# print('k: %s' % k)
	# print('det: %s' % c)

	return 0.5 * (a + b - k + c)


def compute_loss(x_dec, x_next_pred_dec, x, x_next,
				 Qz, Qz_next_pred,
				 Qz_next):
	# Debugging
	# print("in compute_loss")
	# print("x_dec {}, x_next_pred_dec {}, x {}, x_next {}".format(\
	# 	  x_dec.size(), x_next_pred_dec.size(), x.size(), x_next.size()))  # torch.Size([128, 17609])
	# Qz  <e2c_controller_cat.NormalDistribution object at 0x7f940bc49320>
	# Reconstruction losses
	if False:
		x_reconst_loss = (x_dec - x_next).pow(2).sum(dim=1)
		x_next_reconst_loss = (x_next_pred_dec - x_next).pow(2).sum(dim=1)
	else:
		x_reconst_loss = -binary_crossentropy(x, x_dec).sum(dim=1)
		x_next_reconst_loss = -binary_crossentropy(x_next, x_next_pred_dec).sum(dim=1)

	logvar = Qz.logsigma.mul(2)
	print("logvar {}".format(logvar))
	KLD_element = Qz.mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
	print("KLD_element")
	print(KLD_element)
	KLD = torch.sum(KLD_element, dim=1).mul(-0.5)
	print("KLD {}".format(KLD))
	# ELBO
	bound_loss = x_reconst_loss.add(x_next_reconst_loss).add(KLD)
	print("bound_loss {}".format(bound_loss))
	kl = KLDGaussian(Qz_next_pred, Qz_next)
	print("kl {}".format(kl))
	return bound_loss.mean(), kl.mean()

# create a class for the dataset
class CarlaData(Dataset):
	'''
	retrieve data based on frame number: camera images (single central.png) & ctv info.npy
	the img size can be customized, check sensor/hud setting of collected data
	'''
	def __init__(self, ds_dir, img_width = 200, img_height=88):
		# find all available data
		self.dir = ds_dir
		self.img_width = img_width
		self.img_height = img_height
		self.dim_img = self.img_width*self.img_height # *3 for RGB channels
		self.dim_img_lat = 100 # TODO: customize
		self.dim_m = 9 # transform (6), velocity(9)
		self.dim_u = 3
		# self.dim_z = self.dim_img_lat + self.dim_m  # done in E2C_cat __init__
		self._process()

	def __len__(self):
		return len(self._processed)

	def __getitem__(self, index):
		return self._processed[index]

	@staticmethod
	def _process_img(img, img_width, img_height):
		'''
		convert color to gray scale
		(check img_size)
		convert image to tensor
		'''
		# use PIL
		return ToTensor()((img.convert('L').
							resize((img_width, img_height))))

	@staticmethod
	def _process_ctv(ctv):
		# convert a numpy array to tensor: c(3), t(6), v(3)    
		return torch.from_numpy(ctv[:3]).float(), torch.from_numpy(ctv[3:]).float()

	def _process(self):
		preprocessed_file = os.path.join(self.dir, 'processed.pkl')
		if not os.path.exists(preprocessed_file):
			print("writing processed.pkl")
			# create data and dump
			imgs = sorted(glob(os.path.join(self.dir,"*.png"))) # sorted by frame numbers
			# shuffle(imgs) # if need randomness
			frame_numbers = [img.split('/')[-1].split('.')[0] for img in imgs]
			processed = []
			for frame_number in frame_numbers[:-1]: # ignore the last frame which does not have next frame
				next_frame_number = "{:08d}".format(int(frame_number)+1)
				# load images
				img = Image.open(os.path.join(self.dir, frame_number+'.png'))
				img_next_dir = os.path.join(self.dir, next_frame_number+'.png')
				if not os.path.exists(img_next_dir):
					# change climate will jump frame number
					break
				img_next = Image.open(img_next_dir)
				# load ctv array, ctv: control transform, velocity <=> u: action, m: measurement
				ctv = np.load(os.path.join(self.dir, frame_number+'_ctv.npy'))
				ctv_next = np.load(os.path.join(self.dir, next_frame_number+'_ctv.npy'))

				u, m = self._process_ctv(ctv)
				u_next, m_next = self._process_ctv(ctv_next)

				processed.append([self._process_img(img, self.img_width, self.img_height), 
								  m, u, 
								  self._process_img(img_next, self.img_width, self.img_height),
								  m_next])                   

			with open(preprocessed_file, 'wb') as f:
				pickle.dump(processed, f)
			self._processed = processed
		else:
			# directly load the pickle file
			with open(preprocessed_file, 'rb') as f:
				self._processed = pickle.load(f)
		shuffle(self._processed)

	def query_data(self):
		if self._processed is None:
			raise ValueError("Dataset not loaded - call CarlaData._process() first.")
		print("_processed length", len(self._processed))
		return list(zip(*self._processed))[0], list(zip(*self._processed))[1], list(zip(*self._processed))[2], list(zip(*self._processed))[3], list(zip(*self._processed))[4]


class CarlaDataPro(Dataset):
	def __init__(self, img, m, u, img_next, m_next):
		self.img = img # list of tensors
		self.m = m
		self.u = u
		self.img_next = img_next
		self.m_next = m_next
		print("tensor type in CarlaDataPro")
		print('img {}, m {}, u {}'.format(self.img[0].size(), self.m[0].size(), self.u[0].size()))

	def __len__(self):
		return len(self.img)

	def __getitem__(self, index):
		# return the item at certain index
		return self.img[index], self.m[index], self.u[index], \
			self.img_next[index], self.m_next[index]

def train(model_path, ds_dir):
	dataset = CarlaData(ds_dir=ds_dir)
	img, m, u, img_next, m_next = dataset.query_data()

	# build network
	train_dataset = CarlaDataPro(img, m, u, img_next, m_next)
	train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)

	model = E2C_cat(dataset.dim_img, dataset.dim_img_lat, dataset.dim_m, dataset.dim_u)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

	epochs = 50
	lat_file = os.path.join(dataset.dir, 'lat.pkl')
	ztv_file = os.path.join(dataset.dir, 'ztv_trans.pkl')

	for epoch in range(epochs):
		model.train()
		train_losses = []

		for i, (img, m, u, img_next, m_next) in enumerate(train_loader):
			img = img.view(img.shape[0], -1)
			m = m.view(m.shape[0], -1)
			u = u.view(u.shape[0], -1)
			img_next = img_next.view(img_next.shape[0], -1)
			m_next = m_next.view(m_next.shape[0], -1)

			optimizer.zero_grad()
			model(img, m, u, img_next, m_next)

			# TODO: should we return two, img_dec and m_dec?
			x =model.cat(img, m)
			x_next = model.cat(img_next, m_next)
			loss, _ = compute_loss(model.x_dec, model.x_next_pred_dec, x, x_next, model.Qz, model.Qz_next_pred, model.Qz_next)
			
			loss.backward()
			optimizer.step()
			train_losses.append(loss.item())
		model.eval()
		print('epoch : {}, train loss : {:.4f}'\
		   .format(epoch+1, np.mean(train_losses)))

	model.eval()
	torch.save(model, model_path)

def test(model_path, ds_dir, mode='control'):
	dataset = CarlaData(ds_dir=ds_dir, mode=mode)
	# test the model and check the image output

	model = torch.load(model_path)
	model.eval()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	ds_dir = ds_dir
	dataset = CarlaData(ds_dir = ds_dir)
	img_width = dataset.img_width
	img_height = dataset.img_height
	print("number of samples {}".format(len(dataset)))
	num_test_samples = 1
	test_samples = random.sample(range(0,len(dataset)-2), num_test_samples)
	print("test_samples", test_samples)

	for i in test_samples:
		img = dataset[i][0]
		m = dataset[i][1]
		u = dataset[i][2]
		img_next = dataset[i][3]
		m_next = dataset[i][4]   

		img = img.view(1, -1)
		m = m.vie(1, -1)
		u = u.view(1, -1)
		img_next = img_next.view(1, -1)
		m_next = m_next.view(1, -1)
		print("img {}, m {}, u {}".format(img.size(), m.size(), u.size()))
		# get latent vector z
		# z = model.latent_embeddings(img, m)

		# get predicted x_next
		img_pred = model.predict(img, m, u)

		# convert x to image
		img = torch.reshape(img,(1, img_height, img_width))
		x_image = F.to_pil_image(img)
		x_image.save("results_of_e2c/gray_scale/x.png", "PNG")
		x_image.show(title='img')

		# convert x_next to image
		img_next = torch.reshape(img_next,(1, img_height, img_width))
		print("after reshape", img_next.size())
		x_image = F.to_pil_image(img_next)
		x_image.save("results_of_e2c/gray_scale/x_next.png", "PNG")
		x_image.show(title='img_next')

		# convert x_pred to image
		img_pred = torch.reshape(img_pred,(1, img_height, img_width))
		x_image = F.to_pil_image(img_pred)
		x_image.save("results_of_e2c/gray_scale/x_pred.png", "PNG")
		x_image.show(title='img_pred')


if __name__ == '__main__':
	# config dataset path
	# ds_dir = '/home/ruihan/scenario_runner/data/' # used to generate E2C_model_basic, image + c
	# ds_dir = '/home/ruihan/scenario_runner/data_mini/' # used to try dynamics model, image + ctv
	ds_dir = '/home/ruihan/scenario_runner/data_ctv_mini/'

	# config model path
	# model_path = 'models/E2C/E2C_model_basic.pth'
	# model_path = 'models/E2C/E2C_model_try.pth'
	model_path = 'models/E2C/E2C_model_ctv_for_cat.pth'
	train(model_path, ds_dir)
	test(model_path, ds_dir)

	# train_dynamics()

