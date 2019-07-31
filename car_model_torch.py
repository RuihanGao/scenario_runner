'''
Train a car dynamics model (two-layer NN) using PyTorch framework
'''

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import csv
import random

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from e2c_NN import ZUData

torch.set_default_dtype(torch.float32)

# build the network
class Dyn_NN(nn.Module):
	def __init__(self, nx=4, ny=2, nh=50):
		super(Dyn_NN, self).__init__()
		self.layers = nn.Sequential(
			nn.Linear(nx, nh),
			nn.ReLU(),
			nn.Linear(nh, ny)
		)

	def forward(self, x):
		x = x.view(x.size(0), -1)
		return self.layers(x)


if __name__ == '__main__':
	# config dataset path

	# Train a dynamics model
	MLP_model_path = 'models/MLP/Dyn_model_NN_SGD.pth'
	MLP_dict_path = MLP_model_path.replace("_model_", "_dict_")

	if_print = True
	# normalize data
	max_pos_val = 500
	max_yaw_val = 180
	max_speed_val = 40

	plot = True
	train = True

	# read and parse csv data file

	# row = list(np.hstack((np.array([control.throttle, control.steer, control.brake]), \  [0-2]
	#                       transform_to_arr(cur_loc),[3-8] np.array([cur_vel.x, cur_vel.y, cur_vel.z]),\ [9-11]
	#                       transform_to_arr(next_loc), [12-17] np.array([next_vel.x, next_vel.y, next_vel.z]), \ [18-20]
	#                       np.array([cur_loc.location.x, cur_loc.location.y])- np.array([cur_wp.transform.location.x, cur_wp.transform.location.y]), \
	#                       future_wps_np.flatten(), \
	#                       np.array([next_loc.location.x, next_loc.location.y])- np.array([cur_wp.transform.location.x, cur_wp.transform.location.y]), np.array(delta_t))))

	result = []
	line_count = 0
	with open("long_states_dt.csv") as csv_file:
		csv_reader = csv.reader(csv_file)
		for row in csv_reader:
			result.append([float(i) for i in row])
			line_count += 1

	print("process {} lines".format(line_count))
	# Note: if shuffle, traj in testing will not be continuous
	# random.shuffle(result) # shuffle in the first dimension only
	result = np.array(result)

	# input
	yaw = (result[:,7]/max_yaw_val).reshape([-1, 1])
	speed = (np.sqrt(result[:,9]**2 + result[:,10]**2)/max_speed_val).reshape([-1, 1])
	command_data = result[:,:2].reshape([-1, 2])
	dt = result[:,-1].reshape([-1, 1])
	input_data = np.concatenate((yaw, speed, command_data, dt), axis=1).astype(np.float32)

	# output
	delta_x = (result[:,12] - result[:,3]).reshape([-1, 1])
	delta_y = (result[:,13] - result[:,4]).reshape([-1, 1])
	speed_next = (np.sqrt(result[:,18]**2 + result[:,19]**2)/max_speed_val).reshape([-1, 1])
	delta_v = (speed_next - speed).reshape([-1, 1])
	delta_theta = (result[:,16] - result[:,7]).reshape([-1, 1])
	output_data = np.concatenate((delta_x, delta_y, delta_v, delta_theta), axis=1).astype(np.float32)

	# split the train and test datasets
	train_ratio = 0.6
	valid_ratio = 0.2
	test_ratio = 1 - train_ratio - valid_ratio
	train_size = int(train_ratio*line_count)
	valid_size = int(valid_ratio*line_count)
	test_size = int(test_ratio*line_count)


	x_train = input_data[0:train_size,:]
	y_train = output_data[0:train_size,:]

	x_valid = input_data[train_size:train_size+valid_size,:]
	y_valid = output_data[train_size:train_size+valid_size,:]

	x_test = input_data[train_size+valid_size: train_size+valid_size+test_size,:]
	y_test = output_data[train_size+valid_size: train_size+valid_size+test_size,:]

	train_dataset = ZUData(z=x_train, u=y_train)
	valid_dataset = ZUData(z=x_valid, u=y_valid)
	test_dataset = ZUData(z=x_test, u=y_test)

	if train:
		gs_log = "models/MLP/" + MLP_model_path.split("/")[-1][:-4] + "_gs_log.csv"
		# with open(gs_log, 'w', newline='') as csvFile:
		# 	writer = csv.writer(csvFile)
		# 	writer.writerow(["lr", "last_train_loss", "last_valid_loss"])
		lr_range = [0.0001, 0.001, 0.01, 0.1, 1]
		batch_range = [16, 32, 64, 128, 256]
		# if conduct grid_search
		# for lr in lr_range: # 0.01 is best
		lr =0.01
		batch_size = 64
		with open(gs_log, 'a+', newline='') as csvFile:
			writer = csv.writer(csvFile)
			writer.writerow(["batch_size", "last_train_loss", "last_valid_loss"])
		# for batch_size in batch_range:

		train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
		valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)
		test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

		model = Dyn_NN(nx=5, ny=4) # nx: yaw, speed, throttle, steer, dt; ny: delta_x, delta_y
		print(model)
		optimizer = torch.optim.SGD(model.parameters(), lr=lr)
		loss_fn = torch.nn.MSELoss()

		# config training & validation
		epochs = 500
		print("iterate over epochs")

		train_loss_ep = []
		valid_loss_ep = []
		for epoch in range(epochs):
			model.train()
			
			train_losses = []
			valid_losses = []

			# train
			for i, (x, y) in enumerate(train_loader):
				
				optimizer.zero_grad()
				outputs = model(x)
				loss = loss_fn(outputs, y)
				loss.backward()
				optimizer.step()
				train_losses.append(loss.item())
			
			model.eval()

			# validate
			with torch.no_grad():
				for i, (x, y) in enumerate(valid_loader):
					outputs = model(x)
					loss = loss_fn(outputs, y)
					valid_losses.append(loss.item())

			mean_train = np.mean(train_losses)
			mean_valid = np.mean(valid_losses)
			print('epoch : {}, train loss : {:.4f}, valid loss : {:.4f}'\
			 .format(epoch+1, mean_train, mean_valid))

			train_loss_ep.append(mean_train)
			valid_loss_ep.append(mean_valid)

		model.eval()
		# torch.save(model, MLP_model_path)
		print("save state_dict")
		torch.save({
					'epoch': epoch,
					'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),
					'loss_fn': loss_fn,
					'train_loss_ep': train_loss_ep,
					'valid_loss_ep':valid_loss_ep
					}, MLP_dict_path)

		print("save the entire model")
		torch.save(model, MLP_model_path)
		# if grid_search, save to logs
		row = [batch_size, np.mean(train_loss_ep[-10:]), np.mean(valid_loss_ep[-10:])]
		with open(gs_log, 'a+') as csvFile:
			writer = csv.writer(csvFile)
			writer.writerow(row)
			csvFile.close()

	# Load for resuming training
	print("load state_dict")
	model = Dyn_NN(nx=5, ny=4)
	optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

	checkpoint = torch.load(MLP_dict_path)
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	epoch = checkpoint['epoch']
	loss = checkpoint['loss_fn']
	train_loss_ep = checkpoint['train_loss_ep']
	valid_loss_ep = checkpoint['valid_loss_ep']

	model.eval()
	# - or -
	# model.train()

	# helper function for plots
	def connectpoints(x,y,i,f=''):
		# x, y are series; p1, p2 are indexer
		x1, x2 = x[i][0], y[i][0]
		y1, y2 = x[i][1], y[i][1]
		plt.plot([x1,x2],[y1,y2],f)

	if plot:
		# plot the loss values
		plt.figure(0)
		plt.plot(train_loss_ep)
		plt.plot(valid_loss_ep)
		plt.xlabel('Epoch number')
		plt.ylabel('Loss')
		plt.legend(['Train', "Validation"], loc='upper left')
		plt.savefig('models/MLP/{}_loss.png'.format(MLP_model_path.split("/")[-1][:-4]))
		# plt.show()

	# plot the predicted trajectories
	# test the model, random choose a sample, and plot the ground truth trajectory, and predicted trajectory
	test_horizon = 5
	# TODO data may be of different episode, resulting in discontinuous  ground truth traj
	test_index = random.randint(0,len(test_dataset)-1-test_horizon)
	test_index = 932
	print("test_index", test_index)
	current_states = []
	pred_states = []
	next_states = []
	for i in range(train_size+valid_size+test_index, train_size+valid_size+test_index+test_horizon):
		ground_truth = result[i]
		current_state = result[i, 3:5].reshape([-1,2]).squeeze()
		current_input = np.concatenate((yaw[i], speed[i], command_data[i], dt[i])).astype(np.float32).reshape([1,-1])
		input_tensor = torch.tensor(current_input)
		current_output = model(input_tensor).data.cpu().numpy()[0]
		pred_state = (current_state + current_output[:2]).reshape([-1,2]).squeeze()
		next_state = result[i, 12:14].reshape([-1,2]).squeeze()
		
		# print("current_state", current_state)
		# print("pred_state", pred_state)
		# print("next_state", next_state)

		# append the state to traj
		current_states.append(current_state)
		pred_states.append(pred_state)
		next_states.append(next_state)

	current_states = np.array(current_states)
	pred_states = np.array(pred_states)
	next_states = np.array(next_states)
	plt.figure(1)
	plt.scatter(current_states[:,0], current_states[:,1],c='b', marker='o', linewidth=2,linestyle='dashed', label='current')
	plt.scatter(next_states[:,0], next_states[:,1],c='k', marker='D',linewidth=2, label='next')
	plt.scatter(pred_states[:,0], pred_states[:,1],c='r', marker='X',linewidth=2, label='pred')
	plt.legend(loc='upper left')

	for i in range(test_horizon):
		connectpoints(current_states,next_states,i, f='k-')
		connectpoints(current_states,pred_states,i, f='r-')

	plt.savefig('models/MLP/{}_pred_traj.png'.format(MLP_model_path.split("/")[-1][:-4]))
	# plt.show()
