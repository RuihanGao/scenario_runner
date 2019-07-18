import os, sys
from os import path 
from glob import glob
from random import shuffle

from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

from PIL import Image
from skimage.transform import resize
from skimage.color import rgb2gray

from tqdm import trange, tqdm
import pickle

from e2c_configs import *

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

# create a class for the dataset
class CarlaData(Dataset):
    '''
    input: camera images (single central.png) & control command in array.npy
    the img size can be customized, check sensor/hud setting of collected data
    output: 3D control command: throttle, steer, brake
    '''
    def __init__(self, dir, img_width = 200, img_height=88, u_dim=3):
        # find all available data
        self.dir = dir
        self.img_width = img_width
        self.img_height = img_height
        self.x_dim = self.img_width*self.img_height*3 # RGB channels 
        self.u_dim = u_dim
        self.z_dim = 100 # TODO: check how to set this dim
        self._process()
        self.valid_size = 1/6

    def __len__(self):
        return len(self._processed)

    def __getitem__(self, index):
        return self._processed[index]

    @staticmethod
    def _process_img(img, img_width, img_height):
        # PIL image convert: https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.convert
        # ToTensor: https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.ToTensor
        return ToTensor()((img.convert('RGB').
                            resize((img_width, img_height))))
    @staticmethod
    def _process_control(control):
        # convert a numpy array to tensor
        return torch.from_numpy(control).float()
    
    def _process(self):
        preprocessed_file = os.path.join(self.dir, 'processed.pkl')
        if not os.path.exists(preprocessed_file):
            # create data and dump
            imgs = sorted(glob(os.path.join(self.dir,"*.png"))) # sorted by frame numbers
            # shuffle(imgs) # if need randomness
            frame_numbers = [img.split('/')[-1].split('.')[0] for img in imgs]
            
            processed = []
            for frame_number in frame_numbers[:-1]: # ignore the last frame which does not have next frame
                next_frame_number = "{:08d}".format(int(frame_number)+1)
                x_val = Image.open(os.path.join(self.dir, frame_number+'.png'))
                u_val = np.load(os.path.join(self.dir, frame_number+'.npy'))
                x_next = Image.open(os.path.join(self.dir, next_frame_number+'.png'))

                processed.append([self._process_img(x_val, self.img_width, self.img_height), 
                                  self._process_control(u_val), 
                                  self._process_img(x_next, self.img_width, self.img_height)])

            with open(preprocessed_file, 'wb') as f:
                pickle.dump(processed, f)
            self._processed = processed
        else:
            # directly load the data
            with open(preprocessed_file, 'rb') as f:
                self._processed = pickle.load(f)
        # TODO: check whether the pickle format it correct (should be tensors)
        shuffle(self._processed)

    def query_data(self):
        if self._processed is None:
            raise ValueError("Dataset not loaded - call CarlaData._process() first.")
        # self.x_val = self._processed[:, 0]
        # self.u_val = self._processed[:, 1]
        # self.x_next = self._processed[:, 2]
        # print("self._processed", len(self._processed))
        # print(self._processed[0])

        return list(zip(*self._processed))[0], list(zip(*self._processed))[1], list(zip(*self._processed))[2]

    def split_dataset(self, batch_size):
        """
        computes (x_t,u_t,x_{t+1}) pair
        returns tuple of 3 ndarrays with shape
        (batch,x_dim), (batch, u_dim), (batch, x_dim)
        # refer to plane_data2.py
        """
        if self._processed is None:
            raise ValueError("Dataset not loaded - call CarlaData._process() first.")
        pass

    #         # copy from __getitem__ in coil_dataset
    #         img_path = os.path.join(self.root_dir,
    #                                 self.sensor_data_names[index].split('/')[-2],
    #                                 self.sensor_data_names[index].split('/')[-1])

    #         img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    #         # Apply the image transformation
    #         if self.transform is not None:
    #             boost = 1
    #             img = self.transform(self.batch_read_number * boost, img)
    #         else:
    #             img = img.transpose(2, 0, 1)

    #         img = img.astype(np.float)
    #         img = torch.from_numpy(img).type(torch.FloatTensor)
    #         img = img / 255.



    # @staticmethod
    # def sample(cls, sample_size, output_dir, step_size = 1,
    #             apply_control=True, num_shards=10):
    #     # interface with Carla to collect data
    #     # see HumanAgent instead of manual_control for reference of parsing sensor data. 
    #     # sample images from manual_control are stored in /out folder, which are kind of spectator view
    #     # TODO

    #     # initialize the environment
    #     # see wether can use manual_control or NPCAgent for collecting data

    #     # agent.run() while saving images and control output


class CarlaDataPro(Dataset):
    def __init__(self, x_val, u_val, x_next):
        # x,val, u_val, x_next all should be in tensor form
        self.x_val = x_val # list of tensors
        self.u_val = u_val
        self.x_next = x_next
        print("tensor type")
        print(type(self.x_val[0]), type(self.u_val[0]), type(self.x_next[0]))
    def __len__(self):
        return len(self.x_val)

    def __getitem__(self, index):
        # return the item at certain index
        # print(type(self.x_val), len(self.x_val)) # <class 'tuple'> 314
        # print(self.x_val[0].size()) #torch.Size([1, 88, 200]) 
        return self.x_val[index], self.u_val[index], self.x_next[index]

def train():
    ds_dir = '/home/ruihan/scenario_runner/data/'
    dataset = CarlaData(dir = ds_dir)
    x_val, u_val, x_next = dataset.query_data()


    # build network
    train_dataset = CarlaDataPro(x_val, u_val, x_next)
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)

    model = E2C(dataset.x_dim, dataset.z_dim, dataset.u_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 50

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for i, (x, action, x_next) in enumerate(train_loader):
            # flatten the input images into a single 784 long vector
            x = x.view(x.shape[0], -1)
            action = action.view(action.shape[0], -1)
            x_next = x_next.view(x_next.shape[0], -1)
            
            optimizer.zero_grad()
            model(x, action, x_next)
            loss, _ = compute_loss(model.x_dec, model.x_next_pred_dec, x, x_next, model.Qz, model.Qz_next_pred, model.Qz_next)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        model.eval()
        print('epoch : {}, train loss : {:.4f}'\
           .format(epoch+1, np.mean(train_losses)))

if __name__ == '__main__':
    # make a funtion to partially test the program
    train()