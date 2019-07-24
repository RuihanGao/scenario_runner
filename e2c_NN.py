# %matplotlilb inline
import numpy as np

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from e2c_controller_cat import CarlaData, CarlaDataPro, binary_crossentropy
from cat_configs import E2C_cat

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
		# TODO: Use sigmoid, tanh, sigmoid on throttle, steer, brake, respectively
		# t, s, b = torch.split(x, (1, 1, 1), dim=1)
		throttle = self.sig(x[:, 0]).view(x.shape[0],-1)
		steer = self.tanh(x[:, 1]).view(x.shape[0],-1)
		brake = self.sig(x[:, 2]).view(x.shape[0],-1)

		# print("tsb", throttle.size(), steer.size(), brake.size())

		return torch.cat([throttle, steer, brake], dim=1)


class FC_coil(nn.Module):
	"""
	copy the full-connectin network from coil, adpted for MLP controller
	"""
	def __init__(self, nx=106, ny=2, nh=53, p=0.2):
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
    weights = Variable(torch.Tensor([1, 10, 0.1])) # .cuda()  
    pct_var = (input-target)**2
    out = pct_var * weights.expand_as(target)
    loss = out.mean() 
    return loss



if __name__ == '__main__':
	ds_dir = '/home/ruihan/scenario_runner/data_ctv_logdepth_norm/'
	MLP_model_path = 'models/MLP/MLP_model_ctv_logdepth_norm_5_WSE_Adam.pth'
	E2C_model_path = 'models/E2C//E2C_model_ctv_logdepth_norm_5.pth'
	dataset = CarlaData(ds_dir=ds_dir)
	img, m, u, img_next, m_next = dataset.query_data()
	u = list(u)

	for i in u:
		i[1] = torch.div(torch.add(i[1],1.0),2.0) # clip steer to [0,1]
		if i[1]<0 or i[1]>1:
			raise ValueError("clipping needed") 
	print("check type", type(img))
	print("img[0]", img[0].size())

	# init models
	e2c = torch.load(E2C_model_path)
	e2c.eval()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# model = MLP_e2c(dataset.dim_z, dataset.dim_u)
	model = FC_coil(dataset.dim_z, dataset.dim_u)

	print("obtain latent_embeddings")
	# TODO try to simplify using map method
	z = []
	for i in range(len(dataset)):
		img_i = img[i].view(1, -1)
		m_i = m[i].view(1, -1)
		z_i = e2c.latent_embeddings(img_i, m_i)
		z.append(z_i)

	train_dataset = ZUData(z=z, u=u)
	train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
	
	optimizer = torch.optim.Adam(model.parameters(), lr=0.0002) # Adam, SGD for MLP, RMS-prop
	# loss_fn = torch.nn.MSELoss() 
	# print("move model to cuda")
	# e2c.cuda()
	# model.cuda()

	# TODO: How to  utilize GPU for training and/or testing

	epochs = 50

	print("iterate over epochs")
	mean_train_losses = []
	for epoch in range(epochs):
		model.train()
		
		train_losses = []
		for i, (z, u) in enumerate(train_loader):
			z = z.view(z.shape[0], -1)
			u = u.view(u.shape[0], -1)
			# if epoch == 0:
			# 	print(u)

			optimizer.zero_grad()
			outputs = model(z)
			# if i%10 == 0:
			# 	print(i, outputs.size(), u.size())
			# loss = loss_fn(outputs, u)
			# loss = -binary_crossentropy(u, outputs).sum(dim=1).mean()
			loss = weighted_mse_loss(outputs, u)
			loss.backward(retain_graph=True)
			optimizer.step()
			
			train_losses.append(loss.item())
				
		model.eval()


		print('epoch : {}, train loss : {:.4f},'\
		 .format(epoch+1, np.mean(train_losses)))

	model.eval()
	torch.save(model, MLP_model_path)