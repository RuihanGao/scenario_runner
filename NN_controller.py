# %matplotlilb inline
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

torch.set_default_dtype(torch.float64)
# create a class for the Dataset
class ControlDS(Dataset):
	def __init__(self, X, y=None):
		self.X = X
		self.y = y

	def __len__(self):
		return len(self.X.index)

	def __getitem__(self, index):
		# return the item at certain index
		location_info = self.X.iloc[index, ].values #.astype(np.float64) #.astype().reshape()
		# do any processing here

		if self.y is not None:
			return location_info, self.y.iloc[index, ].values #.astype(np.float64)
		return location_info

# create a class for multi-layer perceptron model
class MLP(nn.Module):
	def __init__(self, nx=10, ny=2):
		'''
		nx: number of input location, lanewidth, waypoint(location&rotation)
		ny: number of output, throttle & steer
		'''
		super(MLP, self).__init__()
		self.layers = nn.Sequential(
			nn.Linear(nx, 100),
			nn.Linear(100, 1000),
			nn.Linear(1000,300),
			nn.Linear(300, 100),
			nn.ReLU(),
			nn.Linear(100, ny)
		)

	def forward(self, x):
		# convert tensor (128, 1, 28, 28) --> (128, 1*28*28) for MNIST image
		# print("x in forward", x, x.size())
		x = x.view(x.size(0), -1)
		# print("in forward", x.size())
		x = self.layers(x)
		return x

if __name__ == '__main__':
	# parse the dataset
	# ds1: 'localization_ds.csv'
	ds_file = 'localization_relative_coords_ds.csv'
	train_df = pd.read_csv(ds_file)
	# don't have test_df yet. create a dummy agent using NN controller in CARLA to test
	print("train data shape: ", train_df.shape)
	# split the dataset for validation: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
	# iloc: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iloc.html
	X_train, X_valid, y_train, y_valid = \
		train_test_split(train_df.iloc[:, :-2].astype(np.float64), train_df.iloc[:, -2:].astype(np.float64), test_size=1/6, random_state=42)

	print('train input shape : ', X_train.shape)
	print('train control command shape : ', y_train.shape)
	print('valid input shape : ', X_valid.shape)
	print('valid control command image : ', y_valid.shape)

	train_dataset = ControlDS(X=X_train, y=y_train)
	valid_dataset = ControlDS(X=X_valid, y=y_valid)
	# transform.ToTensor()?

	train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
	valid_loader = DataLoader(dataset=valid_dataset, batch_size=128, shuffle=False)

	# python iterator, similar to for loop
	# use `next` to get one batch of data
	dataiter = iter(train_loader)
	# print("dataiter", len(dataiter)) # 102
	locations, controls = dataiter.next()
	# dataiter.next(): torch.Size([128, 10])

	print('location info shape on PyTroch : ', locations.size())
	print('control commands shape on PyTroch : ', controls.size())

	# build the model
	model = MLP()
	print(model)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
	# loss_fn = nn.CrossEntropyLoss()
	# def loss_fn(outputs, controls):
	#	# get error "RuntimeError: bool value of Tensor with more than one value is ambiguous"
	# 	print(outputs.size(), controls.size())
	# 	return nn.MultiLabelMarginLoss(outputs, controls)
	# loss_fn = nn.MultiLabelMarginLoss() 
	loss_fn = torch.nn.MSELoss() 

	# config training & validation
	mean_train_losses = []
	mean_valid_losses = []
	valid_acc_list = []
	epochs = 50
	err = []
	for epoch in range(epochs):
		model.train()
		
		train_losses = []
		valid_losses = []
		# print("train_loader", len(train_loader))
		for i, (locations, controls) in enumerate(train_loader):
			
			optimizer.zero_grad()
			outputs = model(locations)
			if i%10 == 0:
				print(i, outputs.size(), controls.size())
			loss = loss_fn(outputs, controls)
			loss.backward()
			optimizer.step()
			
			train_losses.append(loss.item())
				
		model.eval()
		correct = 0
		total = 0
		# print("valid_loader", len(valid_loader))
		print_op = True
		with torch.no_grad():
			for i, (locations, controls) in enumerate(valid_loader):
				# print(i)
				outputs = model(locations)
				if i%10 == 0:
					print(i, outputs.size(), controls.size())
				# if print_op:
					# print("outputs from the model")
					# print(outputs)
				loss = loss_fn(outputs, controls)
				
				valid_losses.append(loss.item())
				
				# _, predicted = torch.max(outputs.data, 1)
				# print("predicted, controls", predicted.shape, controls.shape)
				# print(predicted[0], controls[0])
				controls = controls.type(torch.DoubleTensor)
				# if_pass = abs(outputs - controls)/controls < 0.10
				# if print_op:
				# 	print("if_pass")
				# 	print(if_pass)
				# 	print_op = False
				# # correct += (predicted == labels).sum().item()
				# correct += (if_pass).sum().item()
				# total += controls.size(0)
				# convert zero element into a small number to avoid inf
				controls[controls == 0] = 1e-7
				current_err = abs(outputs - controls) # a tensor of size [128,2]
				# print(outputs[-1], controls[-1], current_err[-1])
				# convert tensor to numpy array
				current_err = current_err.data.cpu().numpy()
				if len(err):
					err = np.concatenate((err, current_err))
				else:
					# shallow copy if err is empty
					err = current_err.view()				
				# print("err", current_err.shape, err.shape)


		# mean_train_losses.append(np.mean(train_losses))
		# mean_valid_losses.append(np.mean(valid_losses))
		
		# accuracy = 100*correct/total
		std = np.std(err, axis=0)
		mean = np.mean(err, axis=0)
		print("mean", mean, "std", std)
		# valid_acc_list.append(accuracy)
		# print('epoch : {}, train loss : {:.4f}, valid loss : {:.4f}, valid acc : {:.2f}%'\
		# 	 .format(epoch+1, np.mean(train_losses), np.mean(valid_losses), accuracy))
		print('epoch : {}, train loss : {:.4f}, valid loss : {:.4f}, valid mean : {}, valid std : {}'\
		 .format(epoch+1, np.mean(train_losses), np.mean(valid_losses), mean, std))


	model.eval()
	torch.save(model, 'models/NN_model_relative_epo50.pth')
	
	# test the model with testset
	# test_preds = torch.LongTensor()

	# for i, images in enumerate(test_loader):
	# 	outputs = model(images)
		
	# 	pred = outputs.max(1, keepdim=True)[1]
	# 	test_preds = torch.cat((test_preds, pred), dim=0)

	# out_df = pd.DataFrame()
	# out_df['ID'] = np.arange(1, len(X_test.index)+1)
	# out_df['label'] = test_preds.numpy()

	# out_df.head()
	# out_df.to_csv('submission.csv', index=None)






