import os, sys
from os import path 

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

from skimage.transform import resize
from skimage.color import rgb2gray

from tqdm import trange, tqdm
import pickle

from configs import load_config

torch.set_default_dtype(torch.float64)

def binary_crossentropy(t, o, eps=1e-8):
    return t * torch.log(o + eps) + (1.0 - t) * torch.log(1.0 - o + eps)


def kl_bernoulli(p, q, eps=1e-8):
    # http://ufldl.stanford.edu/tutorial/unsupervised/Autoencoders/
    kl = p * torch.log((p + eps) / (q + eps)) + \
         (1 - p) * torch.log((1 - p + eps) / (1 - q + eps))
    return kl.mean()


class NormalDistribution(object):
    # RH: TODO: check the content
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


# create a class for the whole network architecture
class E2C(nn.Module):
    # RH: see Fig 1, the information flow. 
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
        # reparameterization trick for inference model
        std = logvar.mul(0.5).exp_()
        self.z_mean = mean
        self.z_sigma = std
        eps = torch.FloatTensor(std.size()).normal_()
        if std.data.is_cuda:
            eps.cuda()
        eps = Variable(eps)
        return eps.mul(std).add_(mean), NormalDistribution(mean, std, torch.log(std))

    def forward(self, x, action, x_next):
        mean, logvar = self.encode(x)
        mean_next, logvar_next = self.encode(x_next)

        z, self.Qz = self.reparam(mean, logvar)
        z_next, self.Qz_next = self.reparam(mean_next, logvar_next)

        self.x_dec = self.decode(z)
        self.x_next_dec = self.decode(z_next)

        self.z_next_pred, self.Qz_next_pred = self.transition(z, self.Qz, action)
        self.x_next_pred_dec = self.decode(self.z_next_pred)

        return self.x_next_pred_dec

    def latent_embeddings(self, x):
        return self.encode(x)[0]

    def predict(self, X, U):
        # RH: relationship with forward? got several lines in common 
        mean, logvar = self.encode(X)
        z, Qz = self.reparam(mean, logvar)
        z_next_pred, Qz_next_pred = self.transition(z, Qz, U)
        return self.decode(z_next_pred)


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
    # Reconstruction losses
    if False:
        x_reconst_loss = (x_dec - x_next).pow(2).sum(dim=1)
        x_next_reconst_loss = (x_next_pred_dec - x_next).pow(2).sum(dim=1)
    else:
        x_reconst_loss = -binary_crossentropy(x, x_dec).sum(dim=1)
        x_next_reconst_loss = -binary_crossentropy(x_next, x_next_pred_dec).sum(dim=1)

    logvar = Qz.logsigma.mul(2)
    KLD_element = Qz.mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element, dim=1).mul(-0.5)

    # ELBO
    bound_loss = x_reconst_loss.add(x_next_reconst_loss).add(KLD)
    kl = KLDGaussian(Qz_next_pred, Qz_next)
    return bound_loss.mean(), kl.mean()

from .configs import load_config

# create a class for the dataset
class CarlaData(Dataset):
    '''
    input: camera images (single central / concatenated cent&L&L&R)
    output: control command
    '''
    def __init__(self, dir):
        self.dir = dir
        self.width = 256*4
        self.height = 256 # TODO: check the img size returned from rgb camera sensor
        action_dim = 2
        with open(path.join(dir, 'data.json')) as f:
            # TODO: check how to load/dump img with json
            self._data = json.load(f)
        self._process()

    def __len__(self):
        return len(self._data['samples'])

    def __getitem__(self, index):
        return self._processed(index)

    @staticmethod
    def _process_img(img):
        # TODO: check skimage package
        return ToTensor()((img.convert('L').
                            resize((self.width, self.height))))

    def _process(self):
        preprocessed_file = os.path.join(self.dir, 'processed.pkl')
        if not os.path.exists(preprocessed_file):
            # create data and dump
            processed = []
            for sample in tqdm(self._data['samples'], desc='processing_data'):
                # parse the image data from three rgb camreas, C, L, R
                central = Image.open(os.path.join(self.dir, sample['central']))
                left = Image.open(os.path.join(self.dir, sample['left']))
                right = Image.open(os.path.join(self.dir, sample['right']))
                # covert to tensors and add to array
                processed.append((self._process_img(central),
                                  self._process_img(left),
                                  self._process_img(right),
                                  np.array(sample['control'])))
            with open(preprocessed_file, 'wb') as f:
                pickle.dump(processed, f)
            self._processed = processed
        else:
            # directly load the data
            with open(preprocessed_file, 'rb') as f:
                self._processed = pickle.load(f)
    @staticmethod
    def sample(cls, sample_size, output_dir, step_size = 1,
                apply_control=True, num_shards=10):
        # interface with Carla to collect data










    def __init__(self,dir):
        # fname is the .npz data file name
        # TODO: to be obtained from data collection
        super(CarlaData, self).__init__()
        self.cache = fname
        self.initialized = False
        self.params = (x_dim, u_dim, T)


    def initialize(self):
        if os.path.exists(self.cache):
            self.load()
        else:
            self.precompute()
        self.initialized = True
    def load(self):
        print("Loading data from %s ..." % (self.cache))
        D = np.load(self.cache) # TODO: check loading format, e.g. imread
        self.













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
