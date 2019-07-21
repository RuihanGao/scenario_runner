import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from e2c_controller_cat import NormalDistribution

torch.set_default_dtype(torch.float32)

'''
modify e2c to concatenate img and m, 
which are taken care of in Encoder_cat, Decoder_cat
'''

class Encoder_cat(nn.Module):
	def __init__(self, dim_in, dim_out, dim_m):
		'''
		dim_in, dim_out: dimensions config for the image encoder
		dim_m: the dimension of measurements that are simply passed through a constant layeer
		'''
		super(Encoder_cat, self).__init__()
		
		self.enc = nn.Sequential(
			nn.Linear(dim_in, 150),
			nn.BatchNorm1d(150),
			nn.ReLU(),
			nn.Linear(150, 150),
			nn.BatchNorm1d(150),
			nn.ReLU(),
			nn.Linear(150, 150),
			nn.BatchNorm1d(150),
			nn.ReLU(),
			nn.Linear(150, dim_out*2) 
		)

		# self.cat = nn.Identity() # a constant layer linear function
		self.cat = nn.Sequential(
			nn.Linear(dim_m, dim_m*2)) # try to cal mean and var

	def forward(self, img, m):
		'''
		separate the module into two parts,
		one is derived from original encoder network to process img x
		the other is a constant linear layer to pass measurement m
		'''
		img_mean, img_var = self.enc(img).chunk(2, dim=1)
		m_mean, m_var = self.cat(m).chunk(2, dim=1)
		# concate the tensors
		mean = torch.cat([img_mean, m_mean], dim=1)
		var = torch.cat([img_var, m_var], dim=1)

		return mean, var # otherwise just return tv


class Decoder_cat(nn.Module):
	def __init__(self, dim_in, dim_out, dim_m):
		super(Decoder_cat, self).__init__()
		self.dim_in = dim_in
		self.dim_out = dim_out
		self.dim_m = dim_m

		self.dec = nn.Sequential(
			nn.Linear(self.dim_in, 200),
			nn.BatchNorm1d(200),
			nn.ReLU(),
			nn.Linear(200, 200),
			nn.BatchNorm1d(200),
			nn.ReLU(),
			nn.Linear(200, self.dim_out),
			nn.BatchNorm1d(self.dim_out), # TODO: why add BN, SM layers compared to encoder?
			nn.Sigmoid()
		)

		self.cat = nn.Identity() # a constant layer linear function

	def forward(self, z):
		# unevenly split the tensor
		z_img, z_m = torch.split(z, (self.dim_in, self.dim_m), dim=1)
		# TODO: concatenate img_dec and m_dec
		return torch.cat([self.dec(z_img), self.cat(z_m)], dim=1)

# copy from e2c_configs
class Transition_cat(nn.Module):
	def __init__(self, dim_z, dim_u):
		'''
		dim_z: total dimension of concatenated img and m
		dim_u: dimension of action
		'''
		super(Transition_cat, self).__init__()
		self.dim_z = dim_z
		self.dim_u = dim_u

		self.trans = nn.Sequential(
			nn.Linear(dim_z, 100),
			nn.BatchNorm1d(100),
			nn.ReLU(),
			nn.Linear(100, 100),
			nn.BatchNorm1d(100),
			nn.ReLU(),
			nn.Linear(100, dim_z*2)
		)

		self.fc_B = nn.Linear(dim_z, dim_z * dim_u)
		self.fc_o = nn.Linear(dim_z, dim_z)

	def forward(self, h, Q, u):
		# h is mean, Q is var, u is action
		batch_size = h.size()[0]
		# torch chunk method: split into 2 chunks
		v, r = self.trans(h).chunk(2, dim=1)
		v1 = v.unsqueeze(2)
		rT = r.unsqueeze(1)
		I = Variable(torch.eye(self.dim_z).repeat(batch_size, 1, 1))
		if rT.data.is_cuda:
			I.dada.cuda()
		A = I.add(v1.bmm(rT))
		B = self.fc_B(h).view(-1, self.dim_z, self.dim_u)
		o = self.fc_o(h)

		# need to compute the parameters for distributions
		# as well as for the samples
		u = u.unsqueeze(2)
		# Q.mu.unsqueeze(2) (z^bar): torch.Size([128, 100, 1])
		d = A.bmm(Q.mu.unsqueeze(2)).add(B.bmm(u)).add(o.unsqueeze(2)).squeeze(2)
		# print("d", d.size()) # ([128, 100])
		sample = A.bmm(h.unsqueeze(2)).add(B.bmm(u)).add(o.unsqueeze(2)).squeeze(2)

		return sample, NormalDistribution(d, Q.sigma, Q.logsigma, v=v, r=r)


class E2C_cat(nn.Module):

	def __init__(self, dim_img, dim_img_lat, dim_m, dim_u):

		super(E2C_cat, self).__init__()
		self.dim_img = dim_img
		self.dim_img_lat = dim_img_lat
		self.dim_m = dim_m
		self.dim_z = self.dim_img_lat + self.dim_m
		self.dim_u = dim_u

		self.encoder = Encoder_cat(self.dim_img, self.dim_img_lat, self.dim_m)
		self.decoder = Decoder_cat(self.dim_img_lat, self.dim_img, self.dim_m)
		self.trans = Transition_cat(self.dim_z, self.dim_u)

	def encode(self, img, m):
		return self.encoder(img, m)  # return two: z_mean, z_var

	def decode(self, z):
		return self.decoder(z)

	def transition(self, z, Qz, u):
		# TODO
		return self.trans(z, Qz, u)

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

	def forward(self, img, m, u, img_next, m_next):
		self.img = img
		self.m = m
		self.u = u
		self.img_next = img_next
		self.m_next = m_next

		mean, logvar = self.encode(img, m)
		mean_next, logvar_next = self.encode(img_next, m_next)

		# simultaneously perform reparameterization on z (combined by img & m)
		self.z, self.Qz = self.reparam(mean, logvar)
		self.z_next, self.Qz_next = self.reparam(mean_next, logvar_next)
		# self.img_dec, self.m_dec = self.decode(self.z)
		self.x_dec = self.decode(self.z)
		# self.img_next_dec, self.m_next_dec = self.decode(self.z_next)
		self.x_next_dec = self.decode(self.z_next)
		self.z_next_pred, self.Qz_next_pred = self.transition(self.z, self.Qz, u)
		# self.img_next_pred_dec, self.m_next_pred_dec = self.decode(self.z_next_pred)
		self.x_next_pred_dec = self.decode(self.z_next_pred)
		# return self.img_next_pred_dec, self.m_next_pred_dec
		return self.x_next_pred_dec

	def latent_embeddings(self, img, m):
		return self.encode(img, m)[0]

	def predict(self, img, m, u):
		mean, logvar = self.encode(img, m)
		z, Qz = self.reparam(mean, logvar)
		z_next_pred, Qz_next_pred = self.transition(z, Qz, u)
		return self.decode(z_next_pred)

	def split_dec(self, z):
		return torch.split(self.decode(z), (self.dim_img, self.dim_m), dim=1) # return img_dec, m_dec
	
	def cat(self, img, m):
		return torch.cat([img, m], dim=1)
