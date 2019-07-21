import os, sys
from os import path 
from glob import glob
import random
from random import shuffle

from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, ToPILImage
import torchvision.transforms.functional as F
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

    def forward(self, x, action, x_next, tv=None, tv_next=None):

        self.x = x
        self.u = action
        mean, logvar = self.encode(x)
        mean_next, logvar_next = self.encode(x_next)

        self.z, self.Qz = self.reparam(mean, logvar) # Q is approximate posterior
        self.z_next, self.Qz_next = self.reparam(mean_next, logvar_next)

        self.x_dec = self.decode(self.z)
        self.x_next_dec = self.decode(self.z_next)

        self.z_next_pred, self.Qz_next_pred = self.transition(self.z, self.Qz, action)
        self.x_next_pred_dec = self.decode(self.z_next_pred)

        if tv is None:
            self.mode = 'control'
            return self.x_next_pred_dec
        else:
            self.mode = 'ctv'
            self.tv = tv
            self.tv_next = tv_next
            return self.x_next_pred_dec

    def latent_embeddings(self, x):
        return self.encode(x)[0]

    def predict(self, X, U): 
        mean, logvar = self.encode(X)
        z, Qz = self.reparam(mean, logvar)
        z_next_pred, Qz_next_pred = self.transition(z, Qz, U)
        return self.decode(z_next_pred)
    
    def save_z(self):
        # save(x, u, z) pairs
        return self.x.data.cpu().numpy(), self.u.data.cpu().numpy(), self.z.data.cpu().numpy()


    def save_ztv(self):
        if self.mode == 'control':
            raise ValueError("No ztv pairs to save in control mode")
        else:
            # TODO check whether should use hstack since dim 0 is batch number 
            z = self.z.data.cpu().numpy()
            tv = self.tv.data.cpu().numpy()
            z_next = self.z_next.data.cpu().numpy()
            tv_next = self.tv_next.data.cpu().numpy()    

            return np.hstack((z,tv)), self.u.data.cpu().numpy(), np.hstack((z_next,tv_next))

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
    def __init__(self, ds_dir, img_width = 200, img_height=88, mode='control'):
        # find all available data
        self.dir = ds_dir
        self.img_width = img_width
        self.img_height = img_height
        self.x_dim = self.img_width*self.img_height#*3 for RGB channels
        self.mode = mode
        # if self.mode == 'ctv':
        #     self.u_dim = 12 # c(3), t(6), v(3)
        # else: # control
        #     self.u_dim = 3
        self.u_dim = 3
        self.z_dim = 100 # TODO: check how to set this dim
        self._process()
        self.valid_size = 1/6

    def __len__(self):
        return len(self._processed)

    def __getitem__(self, index):
        return self._processed[index]

    @staticmethod
    def _process_img(img, img_width, img_height):
        '''
        convert color to gray scale
        (check img_size)
        convert image to tensor
        '''
        # use PIL
        return ToTensor()((img.convert('L').
                            resize((img_width, img_height))))

    @staticmethod
    def _process_control(control):
        # convert a numpy array to tensor
        return torch.from_numpy(control).float()

    @staticmethod
    def _process_ctv(ctv):
        # c(3), t(6), v(3)    
        return torch.from_numpy(ctv[:3]).float(), torch.from_numpy(ctv[3:]).float()


    def _process(self):
        preprocessed_file = os.path.join(self.dir, 'processed.pkl')
        if not os.path.exists(preprocessed_file):
            print("writing processed.pkl")
            # create data and dump
            imgs = sorted(glob(os.path.join(self.dir,"*.png"))) # sorted by frame numbers
            # shuffle(imgs) # if need randomness
            frame_numbers = [img.split('/')[-1].split('.')[0] for img in imgs]
            processed = []
            for frame_number in frame_numbers[:-1]: # ignore the last frame which does not have next frame
                next_frame_number = "{:08d}".format(int(frame_number)+1)
                # use PIL
                x = Image.open(os.path.join(self.dir, frame_number+'.png'))
                if self.mode == 'control':
                    u = np.load(os.path.join(self.dir, frame_number+'.npy'))
                elif self.mode == 'ctv':
                    u = np.load(os.path.join(self.dir, frame_number+'_ctv.npy'))
                    u_next = np.load(os.path.join(self.dir, next_frame_number+'_ctv.npy'))
                else:
                    raise ValueError("no suitable mode to load u data")
                x_next_dir = os.path.join(self.dir, next_frame_number+'.png')
                if not os.path.exists(x_next_dir):
                    # change climate will jump frame number
                    break
                x_next = Image.open(x_next_dir)
                if self.mode == 'control':
                    processed.append([self._process_img(x, self.img_width, self.img_height), 
                                      self._process_control(u), 
                                      self._process_img(x_next, self.img_width, self.img_height)])
                elif self.mode == 'ctv':
                    u, tv = self._process_ctv(u)
                    u_next, tv_next = self._process_ctv(u_next)

                    processed.append([self._process_img(x, self.img_width, self.img_height), 
                                      tv, u, 
                                      self._process_img(x_next, self.img_width, self.img_height),
                                      tv_next])                   
                else:
                    raise ValueError("no suitable mode to load u data")
            
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
        print("_processed length", len(self._processed))
        if self.mode == 'control': # x, u, x_next
            return list(zip(*self._processed))[0], list(zip(*self._processed))[1], list(zip(*self._processed))[2]
        elif self.mode == 'ctv': # x, tv, u, x_next, tv_next
            return list(zip(*self._processed))[0], list(zip(*self._processed))[1], list(zip(*self._processed))[2], list(zip(*self._processed))[3], list(zip(*self._processed))[4]


class CarlaDataPro(Dataset):
    def __init__(self, x_val, u_val, x_next, tv=None, tv_next=None):
        # x,val, u_val, x_next all should be in tensor form
        self.x_val = x_val # list of tensors
        self.u_val = u_val
        self.x_next = x_next
        if tv is None:
            self.mode = 'control'
            print("tensor type")
            print(type(self.x_val[0]), type(self.u_val[0]), type(self.x_next[0]))
        else:
            self.mode = 'ctv'
            self.tv = tv
            self.tv_next = tv_next
            print("tensor type")
            print(type(self.x_val[0]), type(self.u_val[0]), type(self.x_next[0]), type(self.tv[0]). type(self.tv_next[0]))

    def __len__(self):
        return len(self.x_val)

    def __getitem__(self, index):
        # return the item at certain index
        # print(type(self.x_val), len(self.x_val)) # <class 'tuple'> 314
        # print(self.x_val[0].size()) #torch.Size([1, 88, 200]) 
        if self.mode == 'control':
            return self.x_val[index], self.u_val[index], self.x_next[index]
        elif self.mode == 'ctv':
            return self.x_val[index], self.u_val[index], self.x_next[index], \
                self.tv[index], self.tv_next[index]

def train(model_path, ds_dir, mode='control'):
    dataset = CarlaData(ds_dir=ds_dir, mode=mode)
    if mode == 'control':

        x_val, u_val, x_next = dataset.query_data()
    elif mode == 'ctv':
        # x, tv, u, x_next, tv_next
        x_val, tv, u_val, x_next, tv_next = dataset.query_data()

    # build network
    if mode == 'control':
        train_dataset = CarlaDataPro(x_val, u_val, x_next)
        train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)

        model = E2C(dataset.x_dim, dataset.z_dim, dataset.u_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    elif mode == 'ctv':
        train_dataset = CarlaDataPro(x_val, u_val, x_next, tv=tv, tv_next=tv_next)
        train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)

        model = E2C(dataset.x_dim, dataset.z_dim, dataset.u_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    epochs = 50
    lat_file = os.path.join(dataset.dir, 'lat.pkl')
    ztv_file = os.path.join(dataset.dir, 'ztv_trans.pkl')

    for epoch in range(epochs):
        model.train()
        train_losses = []

        if mode == 'control':
            for i, (x, u, x_next) in enumerate(train_loader):
                # flatten the input images into a single 784 long vector
                x = x.view(x.shape[0], -1)
                u = u.view(u.shape[0], -1)
                x_next = x_next.view(x_next.shape[0], -1)
                
                optimizer.zero_grad()
                model(x, u, x_next)
                
                if epoch == epochs -1: # save latent vector during last epoch
                    x_save, u_save, z_save = model.save_z()
                    lat = [x_save, u_save, z_save]
                    with open(lat_file, 'wb') as f:
                        # TODO: check whehter it is appending
                        pickle.dump(lat, f)

                loss, _ = compute_loss(model.x_dec, model.x_next_pred_dec, x, x_next, model.Qz, model.Qz_next_pred, model.Qz_next)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            model.eval()
            print('epoch : {}, train loss : {:.4f}'\
               .format(epoch+1, np.mean(train_losses)))

        elif mode == 'ctv':
            for i, (x, u, x_next, tv, tv_next) in enumerate(train_loader):
                x = x.view(x.shape[0], -1)
                u = u.view(u.shape[0], -1)
                x_next = x_next.view(x_next.shape[0], -1)
                tv = tv.view(tv.shape[0], -1)
                tv_next = tv_next.view(tv_next.shape[0], -1)

                optimizer.zero_grad()
                model(x, u, x_next, tv=tv, tv_next=tv_next)

                if epoch == epochs -1: # save latent vector during last epoch
                    x_save, u_save, z_save = model.save_z()
                    lat = [x_save, u_save, z_save]
                    with open(lat_file, 'wb') as f:
                        # TODO: check whehter it is appending
                        pickle.dump(lat, f)

                    # for ctv mode, save ztv transition
                    print("saving ztv")
                    ztv_save, u_save, ztv_next_save = model.save_ztv()
                    ztv = [ztv_save, u_save, ztv_next_save]
                    with open(ztv_file, 'wb') as f:
                        # TODO: check whehter it is appending
                        pickle.dump(ztv, f)

                loss, _ = compute_loss(model.x_dec, model.x_next_pred_dec, x, x_next, model.Qz, model.Qz_next_pred, model.Qz_next)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            model.eval()
            print('epoch : {}, train loss : {:.4f}'\
               .format(epoch+1, np.mean(train_losses)))

    model.eval()
    torch.save(model, model_path)

def test(model_path, ds_dir, mode='control'):
    dataset = CarlaData(ds_dir=ds_dir, mode=mode)
    # test the model and check the image output

    model = torch.load(model_path)
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ds_dir = ds_dir
    dataset = CarlaData(ds_dir = ds_dir)
    img_width = dataset.img_width
    img_height = dataset.img_height
    print("number of samples {}".format(len(dataset)))
    num_test_samples = 1
    test_samples = random.sample(range(0,len(dataset)-2), num_test_samples)
    print("test_samples", test_samples)

    for i in test_samples:
        if mode == 'control':
            x = dataset[i][0]
            u = dataset[i][1]
            x_next = dataset[i][2]

        else:
            # in 'ctv' mode, x_val, tv, u_val, x_next, tv_next = dataset.query_data()
            x = dataset[i][0]
            u = dataset[i][2]
            x_next = dataset[i][3]    

        x = x.view(1, -1)
        u = u.view(1, -1)
        x_next = x_next.view(1, -1)
        print("x {}, u {}, x_next {}".format(x.size(), u.size(), x_next.size()))
        # get latent vector z

        # get predicted x_next
        x_pred = model.predict(x, u)

        # compare with true x_next

        # use PIL
        # convert x to image
        x = torch.reshape(x,(1, img_height, img_width))
        x_image = F.to_pil_image(x)
        x_image.save("results_of_e2c/gray_scale/x.png", "PNG")
        x_image.show(title='x')

        # convert x_next to image
        x_next = torch.reshape(x_next,(1, img_height, img_width))
        print("after reshape", x_next.size())
        x_image = F.to_pil_image(x_next)
        x_image.save("results_of_e2c/gray_scale/x_next.png", "PNG")
        x_image.show(title='x_next')

        # convert x_pred to image
        x_pred = torch.reshape(x_pred,(1, img_height, img_width))
        x_image = F.to_pil_image(x_pred)
        x_image.save("results_of_e2c/gray_scale/x_pred.png", "PNG")
        x_image.show(title='x_pred')

def train_dynamics():
    '''
    load (ztv, u, ztv_next) pairs from the dataset (iteratively pickle load)
    build a model to approximate dynamics, similar to transion model in e2c
    '''

    pass
    
if __name__ == '__main__':
    # config dataset path
    # ds_dir = '/home/ruihan/scenario_runner/data/' # used to generate E2C_model_basic, image + c
    # ds_dir = '/home/ruihan/scenario_runner/data_mini/' # used to try dynamics model, image + ctv
    ds_dir = '/home/ruihan/scenario_runner/data_ctv_mini/'
    mode='ctv'

    # config model path
    # model_path = 'models/E2C/E2C_model_basic.pth'
    # model_path = 'models/E2C/E2C_model_try.pth'
    model_path = 'models/E2C/E2C_model_ctv.pth'
    # train(model_path, ds_dir, mode)
    test(model_path, ds_dir, mode)

    # train_dynamics()

