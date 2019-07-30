import os, sys
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import csv

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from NN_controller import ControlDS
from e2c_NN import ZUData

torch.set_default_dtype(torch.float32)

# build the network
class Dyn_NN(nn.Module):
	def __init__(self, nx=4, ny=2, nh=50):
		super(Dyn_NN, self).__init__()
		self.layers = nn.Sequential(
			nn.Linear(nx, nh),
			nn.Sigmoid(),
			nn.Linear(nh, ny)
		)

	def forward(self, x):
		x = x.view(x.size(0), -1)
		return self.layers(x)


if __name__ == '__main__':
	# config dataset path

	# Train a dynamics model
	MLP_model_path = 'models/MLP/Dyn_NN_SGD.pth'

	train_size = 20000
	test_size = 3000

	x = []
	y = []
	line_count = 0	
	if_print = True
	max_pos_val = 500
	max_yaw_val = 180
	max_speed_val = 40

	result = []
	plot = True

	# read and parse csv data file
	with open("long_states_2.csv") as csv_file:
		csv_reader = csv.reader(csv_file)
	
		# convert to float format
		for row in csv_reader:
			result.append([float(i) for i in row])
	# # subtract to get delta_t
	# for i in range(len(result)-1):
	# 	result[i][0] = result[i+1][0] - result[i][0]
	result = np.array(result)

	# px = row[3]
	# py = row[4]
	# speed = math.sqrt(row[9]**2+row[10]**2)
	# yaw = row[7]
	yaw = (result[:,7]/max_yaw_val).reshape([-1, 1])
	speed = (np.sqrt(result[:,9]**2 + result[:,10]**2)/max_speed_val).reshape([-1, 1])
	command_data = result[:,:2].reshape([-1, 2])
	input_data = np.concatenate((yaw, speed, command_data), axis=1).astype(np.float32)

	delta_x = np.zeros((len(result)-1,1))
	delta_y = np.zeros((len(result)-1,1))

	for i in range(0, len(result)-1):
	    delta_x[i][0] = result[i+1][3] - result[i][3]
	    delta_y[i][0] = result[i+1][4] - result[i][4]
	output_data = np.concatenate((delta_x, delta_y), axis=1).astype(np.float32)

	# split the train and test datasets

	x_data = input_data[0:train_size,:]
	y_data = output_data[0:train_size,:]

	test_x_data = input_data[train_size:train_size+test_size,:]
	test_y_data = output_data[train_size:train_size+test_size,:]


	train_dataset = ZUData(z=x_data, u=y_data)
	valid_dataset = ZUData(z=test_x_data, u=test_y_data)

	train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
	valid_loader = DataLoader(dataset=valid_dataset, batch_size=128, shuffle=False)
	
	model = Dyn_NN(nx=4, ny=2) # nx: yaw, speed, throttle, steer; ny: delta_x, delta_y
	print(model)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
	loss_fn = torch.nn.MSELoss()

	# config training & validation
	epochs = 500
	print("iterate over epochs")
	mean_train_losses = []
	mean_valid_losses = []

	loss_values = []
	for epoch in range(epochs):
		model.train()
		
		train_losses = []
		valid_losses = []
		running_loss =0.0

		# train
		for i, (x, y) in enumerate(train_loader):
			
			optimizer.zero_grad()
			outputs = model(x)
			loss = loss_fn(outputs, y)
			loss.backward()
			optimizer.step()
			
			train_losses.append(loss.item())
			running_loss += loss.item()
		loss_values.append(running_loss/len(train_dataset))
		model.eval()

		# validata
		with torch.no_grad():
			for i, (x, y) in enumerate(valid_loader):
				outputs = model(x)
				loss = loss_fn(outputs, y)
				valid_losses.append(loss.item())


		print('epoch : {}, train loss : {:.4f}, valid loss : {:.4f}'\
		 .format(epoch+1, np.mean(train_losses), np.mean(valid_losses)))

	model.eval()
	# torch.save(model, MLP_model_path)
	print("save state_dict")
	torch.save({
	            'epoch': epoch,
	            'model_state_dict': model.state_dict(),
	            'optimizer_state_dict': optimizer.state_dict(),
	            'loss': loss_fn
	            }, MLP_model_path)


	# Load for resuming training
	print("load state_dict")
	model = Dyn_NN(nx=4, ny=2)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

	checkpoint = torch.load(MLP_model_path)
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	epoch = checkpoint['epoch']
	loss = checkpoint['loss']

	model.eval()
	# - or -
	# model.train()
	plt.plot(loss_values)
	plt.xlabel('Epoch number')
	plt.ylabel('Train loss')
	plt.savefig('models/MLP/{}_loss.png'.format(MLP_model_path.split("/")[-1][:-4]))
	plt.show()

	
