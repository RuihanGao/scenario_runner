import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from e2c_controller import NormalDistribution
from e2c_configs import Transition

torch.set_default_dtype(torch.float32)

class DynamicsTransition(Transition):
	def __init__(self,dim_z, dim_u):
        trans = nn.Sequential(
            nn.Linear(dim_z, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, dim_z*2)
        )
        super(CarlaTransition, self).__init__(trans, dim_z, dim_u)


class dynamics_mod(nn.module):
	'''
	create a model similar to Transition module in E2c 
	to approximate the dynamics (ztv_{t+1} | ztv_t, u_t)
	'''
	def __init__(self, dim_in, dim_u):
		super(dynamics_mod, self).__init__()
		self.trans = DynamicsTransition(dim_in, dim_u)
	   
	def reparam(self, mean, logvar):
        # reparameterization trick for inference model
        std = logvar.mul(0.5).exp_()
        self.z_mean = mean
        self.z_sigma = std
        eps = torch.FloatTensor(std.size()).normal_()
        if std.data.is_cuda:
            eps.cuda()
        eps = Variable(eps)
        return eps.mul(std).add_(mean), NormalDistribution(mean, std, torch.log(std))
	
	def forward(self, ztv, action, ztv_next):
		self.u = action

		# calculate mena and logvar
		# Mean
		mean = torch.mean(ztv, 1)         # Size 3: Mean in dim 1
		var = torch.var(x.view(ztv.shape[0], x.shape[1], 1, -1,), dim=3, keepdim=True)

		self.ztv = ztv