# %%
import time
import numpy as np
import pickle
from numpy.linalg import det

import CMI_estimation.CMINE_lib as CMINE
# from Guassian_variables import Data_guassian

import pandas as pd
from scipy.stats import multivariate_normal
import itertools

np.random.seed(37)

import numpy as np
from scipy import stats

import numpy as np
from sklearn.neighbors import KernelDensity

# %%


# %%
import numpy as np
import math

import torch 
import torch.nn as nn

# %%
def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


class L1OutUB(nn.Module):  # naive upper bound
    def __init__(self, x_dim, y_dim, hidden_size):
        super(L1OutUB, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def forward(self, x_samples, y_samples): # x_samples = s_t, a ; y_samples = s_{t+1}
        batch_size = y_samples.shape[0]
        #x_samples[:, 1].masked_fill_(x_samples[:, 1]!=0, float(0))
        mu, logvar = self.get_mu_logvar(x_samples)

        positive = (- (mu - y_samples)**2 /2./logvar.exp() - logvar/2.).sum(dim = -1) #[nsample]

        negative = []
        for i in range(x_samples.shape[1]-1):
            x_temp = x_samples.index_fill_(1, torch.tensor([i]).cuda(), float('0'))
            mu, logvar = self.get_mu_logvar(x_temp)
            neg = (- (mu - y_samples)**2 /2./logvar.exp() - logvar/2.).sum(dim = -1) #[nsample]
            if i == 0:
                negative = neg.unsqueeze(-1)
            else:
                negative = torch.cat([negative, neg.unsqueeze(-1)], 1)



        # mu_1 = mu.unsqueeze(1)          # [nsample,1,dim]
        # logvar_1 = logvar.unsqueeze(1)
        # y_samples_1 = y_samples.unsqueeze(0)            # [1,nsample,dim]
        # all_probs =  (- (y_samples_1 - mu_1)**2/2./logvar_1.exp()- logvar_1/2.).sum(dim = -1)  #[nsample, nsample]

        # diag_mask =  torch.ones([batch_size]).diag().unsqueeze(-1).cuda() * (-20.)
        # negative = log_sum_exp(all_probs + diag_mask,dim=0) - np.log(batch_size-1.) #[nsample]
        #print(( positive.unsqueeze(-1)- negative ).mean())
       
        return ( positive.unsqueeze(-1)- negative ).mean()
    '''
    def forward(self, x_samples, y_samples): # x_samples = s_t, a ; y_samples = s_{t+1}
        batch_size = y_samples.shape[0]
        #x_samples[:, 1].masked_fill_(x_samples[:, 1]!=0, float(0))
        mu, logvar = self.get_mu_logvar(x_samples)

        positive = (- (mu - y_samples)**2 /2./logvar.exp() - logvar/2.).sum(dim = -1) #[nsample]



        mu_1 = mu.unsqueeze(1)          # [nsample,1,dim]
        logvar_1 = logvar.unsqueeze(1)
        y_samples_1 = y_samples.unsqueeze(0)            # [1,nsample,dim]
        all_probs =  (- (y_samples_1 - mu_1)**2/2./logvar_1.exp()- logvar_1/2.).sum(dim = -1)  #[nsample, nsample]

        diag_mask =  torch.ones([batch_size]).diag().unsqueeze(-1).cuda() * (-20.)
        negative = log_sum_exp(all_probs + diag_mask,dim=0) - np.log(batch_size-1.) #[nsample]

       
        return (positive - negative).mean()
    '''
        
        
    def loglikeli(self, x_samples, y_samples):
        x_samples = x_samples.clone()
        y_samples = y_samples.clone()
        mu, logvar = self.get_mu_logvar(x_samples)

        lg = (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)

        del x_samples, y_samples
        torch.cuda.empty_cache()
        #print("lg", lg)
        return lg
    
    def loglikeli_mask(self, x_samples, y_samples):
        negative = []
        x_samples = x_samples.clone()
        y_samples = y_samples.clone()
        for i in range(x_samples.shape[1]-1):
            x_temp = x_samples.index_fill_(1, torch.tensor([i]).cuda(), float(0)).clone()
            mu, logvar = self.get_mu_logvar(x_temp)
            neg =  (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=-1) #(- (mu - y_samples)**2 /2./logvar.exp() - logvar/2.).sum(dim = -1) #[nsample]
            if i == 0:
                negative = neg.unsqueeze(-1)
            else:
                negative = torch.cat([negative, neg.unsqueeze(-1)], 1)
        del x_samples, y_samples
        torch.cuda.empty_cache()
        #print('mask', negative.sum(dim=1).mean(dim=0))
        return negative.sum(dim=1).mean(dim=0)

    def learning_loss(self, x_samples, y_samples):
        return  - self.loglikeli_mask(x_samples, y_samples)  - self.loglikeli(x_samples, y_samples)

# %%
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time

#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# %%
def sample_correlated_gaussian(rho=0.5, dim=20, batch_size=128, to_cuda=False, cubic = False):
    """Generate samples from a correlated Gaussian distribution."""
    mean = [0,0]
    cov = [[1.0, rho],[rho, 1.0]]
    x, y = np.random.multivariate_normal(mean, cov, batch_size * dim).T
    #x, y = np.random.multivariate_normal(mean, cov, batch_size * dim).T

    x = x.reshape(-1, dim)
    y = y.reshape(-1, dim)

    if cubic:
        y = y ** 3

    if to_cuda:
        x = torch.from_numpy(x).float().cuda()
        #x = torch.cat([x, torch.randn_like(x).cuda() * 0.3], dim=-1)
        y = torch.from_numpy(y).float().cuda()
    return x, y

# %%
def rho_to_mi(rho, dim):
    result = -dim / 2 * np.log(1 - rho **2)
    return result


def mi_to_rho(mi, dim):
    result = np.sqrt(1 - np.exp(-2 * mi / dim))
    return result

# %%
sample_dim = 20
batch_size = 64
hidden_size = 15
learning_rate = 0.005
training_steps = 4000

cubic = False 

# %%
model = L1OutUB(sample_dim, sample_dim, hidden_size).cuda()
optimizer = torch.optim.Adam(model.parameters(), learning_rate)

# %%
rho = mi_to_rho(6, sample_dim)

# %%
mi_est_values = []

# %%
for step in range(training_steps):
    batch_x, batch_y = sample_correlated_gaussian(rho, dim=sample_dim, batch_size = batch_size, to_cuda = True, cubic = cubic)


    model.eval()
    mi_est_values.append(model(batch_x, batch_y).item())
    # %%
    model.train() 

    model_loss = model.learning_loss(batch_x, batch_y)

    optimizer.zero_grad()
    model_loss.backward(retain_graph=True)
    optimizer.step()

    del batch_x, batch_y
    torch.cuda.empty_cache()
print("finish training for %s with true MI value = %f"%('LOO', 6.0))
print(np.array(mi_est_values).mean())
# %%



