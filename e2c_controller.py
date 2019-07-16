import torch
from torch import nn
from torch.autograd import Variable

from configs import load_config

class E2C(nn.Module):
    def __init__(self, dim_in, dim_z, dim_u, config='carla'):
        super(E2C, self).__init__()
        enc, trans, dec = load_config(config)
        self.encoder = enc(dim_in, dim_z)

        self.decoder = dec(dim_z, dim_in)
        self.trans = trans(dim_z, dim_u)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def transition(self, z, Qz, u):
        return self.trans(z, Qz, u)

    def reparam(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        self.z_mean = mean
        self.z_sigma = std
        eps = torch.FloatTensor(std.size()).normal_()
        if std.data.is_cuda:
            eps.cuda()
        eps = Variable(eps)
        return eps.mul(std).add_(mean), NormalDistribution(mean, std, torch.log(std))