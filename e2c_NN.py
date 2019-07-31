'''
Train a NN controller
'''
import os, sys
import numpy as np
import math

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from e2c_controller_cat import CarlaData, CarlaDataPro, binary_crossentropy
from cat_configs import E2C_cat

import csv
import math
import matplotlib.pyplot as plt 


torch.set_default_dtype(torch.float32)
# torch.set_default_tensor_type('torch.cuda.FloatTensor')

# create a class for the Dataset
class ZUData(Dataset):
	def __init__(self, z, u=None):
		self.z = z
		self.u = u

	def __len__(self):
		return len(self.z)

	def __getitem__(self, index):
		# return the item at certain index
		return self.z[index], self.u[index]


class MLP_e2c(nn.Module):
	def __init__(self, nx=10, ny=3):
		'''
		nx: number of input location, lanewidth, waypoint(location&rotation)
		ny: number of output, throttle & steer
		'''
		super(MLP_e2c, self).__init__()
		self.layers = nn.Sequential(
			nn.Linear(nx, 100),
			nn.Linear(100, 1000),
			nn.Linear(1000,300),
			nn.Linear(300, 100),
			nn.ReLU(),
			nn.Linear(100, ny)
		)
		self.sig = nn.Sigmoid()
		self.tanh = nn.Tanh()

	def forward(self, x):
		# convert tensor (128, 1, 28, 28) --> (128, 1*28*28) for MNIST image
		# print("x in forward", x, x.size())
		x = x.view(x.size(0), -1)
		# print("in forward", x.size())
		x = self.layers(x)
		return self.tanh(x)
		# TODO: Use sigmoid, tanh, sigmoid on throttle, steer, brake, respectively
		# t, s, b = torch.split(x, (1, 1, 1), dim=1)
		# throttle = self.sig(x[:, 0]).view(x.shape[0],-1)
		# steer = self.tanh(x[:, 1]).view(x.shape[0],-1)
		# brake = self.sig(x[:, 2]).view(x.shape[0],-1)

		# print("tsb", throttle.size(), steer.size(), brake.size())

		# return torch.cat([throttle, steer, brake], dim=1)


class FC_coil(nn.Module):
	"""
	copy the full-connectin network from coil, adpted for MLP controller
	"""
	def __init__(self, nx=106, ny=2, nh=53, p=0.1):
		"""
		original coil (512-256-3)
		input: latent_embeddings dim_z = 106
		one hidden layer: 64
		output: dim_u = 3
		p: possibility for dropout
		"""
		super(FC_coil, self).__init__()
		self.layers = nn.Sequential(
			nn.Linear(nx, nh),
			nn.Dropout2d(p=p),
			nn.ReLU(),
			nn.Linear(nh, ny),
			nn.Dropout2d(p=p)
		)
		self.sig = nn.Sigmoid()
		self.tanh = nn.Tanh()

	def forward(self, x):
		x = x.view(x.size(0), -1)
		x = self.layers(x)

		# throttle = self.sig(x[:, 0]).view(x.shape[0],-1)
		# steer = self.tanh(x[:, 1]).view(x.shape[0],-1)
		# brake = self.sig(x[:, 2]).view(x.shape[0],-1)

		# return torch.cat([throttle, steer, brake], dim=1)
		return self.sig(x)



def weighted_mse_loss(input,target):
	#alpha of 0.5 means half weight goes to first, remaining half split by remaining 15
	weights = Variable(torch.Tensor([1000, 0.1])) # .cuda()   # change [1, 1000, 0.1] to [1000, 0.1]
	pct_var = (input-target)**2
	out = pct_var * weights.expand_as(target)
	loss = out.mean() 
	return loss



if __name__ == '__main__':
	# Method 1: For e2c with image input, load from npy data with frame_number
	# num_wps = 50
	# ds_dir = '/home/ruihan/scenario_runner/data_ctv_logdepth_norm_catwp_{}/'.format(num_wps)
	# MLP_model_path = 'models/MLP/MLP_model_ctv_logdepth_norm_catwp_{}_5_WSE_Adam_monly_vy.pth'.format(num_wps)
	# # E2C_model_path = 'models/E2C/E2C_model_ctv_logdepth_norm_5.pth'
	# dataset = CarlaData(ds_dir=ds_dir, num_wps=num_wps)
	# img, m, u, img_next, m_next = dataset.query_data()
	# u = list(u)

	# for i in u:
	# 	i[1] = torch.div(torch.add(i[1],1.0),2.0) # clip steer to [0,1]
	# 	if i[1]<0 or i[1]>1:
	# 		raise ValueError("clipping needed") 
	# print("check type", type(img))
	# print("img[0]", img[0].size())

	# # init models
	# # disable the image part for now
	# # e2c = torch.load(E2C_model_path)
	# # e2c.eval()
	# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# # model = MLP_e2c(dataset.dim_z, dataset.dim_u)
	# # model = FC_coil(dataset.dim_z, dataset.dim_u)
	# model = FC_coil(dataset.dim_m-1, dataset.dim_u) # vx, vy, vz -> yaw, speed, so minus one

	# print("obtain latent_embeddings")
	# # TODO try to simplify using map method
	# z = []
	# for i in range(len(dataset)):
	# 	# img_i = img[i].view(1, -1)
	# 	m_i = m[i].view(1, -1).data.numpy()[0]
	# 	yaw = math.atan2(m_i[-2], m_i[-3])
	# 	speed = np.sqrt(m_i[-2]**2 + m_i[-3]**2 )
	# 	# z_i = e2c.latent_embeddings(img_i, m_i)
	# 	# z.append(z_i)
	# 	mt = np.hstack((m_i[:-3], np.array([yaw, speed]))).astype(np.float32)
	# 	z.append(torch.from_numpy(mt))

	# Method 2: For pure state input, load data from "long_states.csv"
	num_wps = 50
	MLP_model_path = 'models/MLP/MLP_model_long_states_{}.pth'.format(num_wps)
	MLP_dict_path = MLP_model_path.replace('_model_', '_dict_')
	print("dict_path", MLP_dict_path)
	# model type
	model = FC_coil(nx=106, ny=3)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

	# check if path exists
	if os.path.exists(MLP_dict_path):
		print("load state_dict")
		datafile = "long_states_retrain.csv"
		#load the existing model and resume training
		checkpoint = torch.load(MLP_dict_path)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		epoch = checkpoint['epoch']
		print("load state_dict, epoch", epoch)
		loss_fn = checkpoint['loss_fn']

	else:
		# create a fresh new model
		print("create a new model")
		datafile = "long_states_2.csv"
		loss_fn = torch.nn.MSELoss() 

	# process data
	max_pos_val = 500
	max_yaw_val = 180
	max_speed_val = 40

	z = []
	u = []
	line_count = 0
	if_print = True

	with open(datafile) as csv_file:
		csv_reader = csv.reader(csv_file)
		for row in csv_reader:
			row = [float(i) for i in row]
			speed = math.sqrt(float(row[9])**2+float(row[10])**2)/max_speed_val
			yaw = row[7]
			wps = row[-106:-2]
			action = row[:3]
			state = [speed, yaw]+wps
			if if_print:
				print("sample state and action")
				print(state)
				print(action)
				if_print = False
			print(action)

			z.append(torch.from_numpy(np.array(state).astype(np.float32)))
			u.append(torch.from_numpy(np.array(action).astype(np.float32)))
			line_count += 1

	print("process {} lines".format(line_count))
	print("z {}, u {}".format(z[0].size(), u[0].size()))
	train_dataset = ZUData(z=z, u=u)
	train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)

	epochs = 500
	print("iterate over epochs")
	loss_values = []
	if_print = True
	for epoch in range(epochs):
		model.train()
		train_losses = []
		running_loss =0.0
		for i, (z, u) in enumerate(train_loader):
			z = z.view(z.shape[0], -1)
			u = u.view(u.shape[0], -1)

			optimizer.zero_grad()
			outputs = model(z)
			if if_print:
				print("ground truth", u[0])
				print("outputs", outputs[0])
				if_print = False

			loss = loss_fn(outputs, u)
			# loss = -binary_crossentropy(u, outputs).sum(dim=1).mean()
			# loss1 = -binary_crossentropy(u[:, 0], outputs[:, 0]).sum().mean()
			# loss2 = weighted_mse_loss(outputs[:, 1:], u[:, 1:])
			# loss = loss1 + loss2
			loss.backward(retain_graph=True)
			optimizer.step()
			
			train_losses.append(loss.item())
			running_loss += loss.item()
		loss_values.append(running_loss/len(train_dataset))
				
		model.eval()

		print('epoch : {}, train loss : {:.4f},'\
		 .format(epoch+1, np.mean(train_losses)))

	model.eval()
	print("save state_dict")
	torch.save({
	            'epoch': epoch,
	            'model_state_dict': model.state_dict(),
	            'optimizer_state_dict': optimizer.state_dict(),
	            'loss_fn': loss_fn
	            }, MLP_dict_path)

	print("save the entire model")
	torch.save(model, MLP_model_path)

	print(model)
	plt.plot(loss_values)
	plt.xlabel('Epoch number')
	plt.ylabel('Train loss')
	plt.savefig('models/MLP/{}_loss.png'.format(MLP_model_path.split("/")[-1][:-4]))
	plt.show()

	# parse csv data
	# row = list(np.hstack((np.array([control.throttle, control.steer, control.brake]), \
	#                       transform_to_arr(cur_loc), np.array([cur_vel.x, cur_vel.y, cur_vel.z]),\
	#                       transform_to_arr(next_loc), np.array([next_vel.x, next_vel.y, next_vel.z]), \
	#                       np.array([cur_loc.location.x, cur_loc.location.y])- np.array([cur_wp.transform.location.x, cur_wp.transform.location.y]), \
	#                       future_wps_np.flatten(), \
	#                       np.array([next_loc.location.x, next_loc.location.y])- np.array([cur_wp.transform.location.x, cur_wp.transform.location.y]))))
	# action 3, cur_loc 6, cur_vel 3, next_loc 6, next_vel 3,  dt 1
	# relative loc 2 + 51*2 + 2