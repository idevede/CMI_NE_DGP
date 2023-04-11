# %%
# %%
import time
import numpy as np
import pickle
from numpy.linalg import det

import CMINE_lib as CMINE
# from Guassian_variables import Data_guassian

import pandas as pd
from scipy.stats import multivariate_normal
import itertools

np.random.seed(37)
from scipy import stats
from sklearn.neighbors import KernelDensity

import math

import torch 
import torch.nn as nn

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

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
        y_dim = 1
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())

        self.p_mu_neg = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))

        self.p_logvar_neg = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())
        self.linear_map = nn.Linear(1, x_dim)


    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
    
    def get_mu_logvar_neg(self, x_samples, i = 0):
          
        mu = self.p_mu_neg(x_samples)
        logvar = self.p_logvar_neg(x_samples)
        return mu, logvar

    def forward(self, x_samples, y_samples): # x_samples = s_t, a ; y_samples = s_{t+1}
        batch_size = y_samples.shape[0]
        #x_samples[:, 1].masked_fill_(x_samples[:, 1]!=0, float(0))
        cmi_dims =[]
        
        x_samples = x_samples.unsqueeze(-1)
        x_samples = self.linear_map(x_samples)

       
        neg_inf = torch.finfo(x_samples.dtype).min  # 负无穷的值，与张量的数据类型匹配


        max_values= torch.mean(x_samples, dim = 1)
        for k in range(y_samples.shape[1]):
            #max_values, max_indices = torch.max(x_samples, dim=1)
            
            mu, logvar = self.get_mu_logvar(max_values)
            

            positive = (- (mu - y_samples[:,k].unsqueeze(-1))**2 /2./logvar.exp() - logvar/2.).sum(dim = -1) #[nsample]

            negative = []
            
            for i in range(x_samples.shape[1]-1):
                x_new = x_samples.clone()
                x_new[:, i, :] = float(0) #neg_inf  # 使用fill_方法替换指定维度为负无穷

                #x_new = x_samples.masked_fill(x_samples.new_zeros(x_samples.size()).bool()[:, i, :], float('-inf'))
                #max_values, max_indices = torch.max(x_new, dim=1)
                max_values = torch.mean(x_new, dim=1)

                #x_temp = x_samples.index_fill_(1, torch.tensor([i]).cuda(), float('-inf'))
                mu, logvar = self.get_mu_logvar_neg(max_values)
                neg = (- (mu - y_samples[:,k].unsqueeze(-1))**2 /2./logvar.exp() - logvar/2.).sum(dim = -1) #[nsample]
                if i == 0:
                    negative = neg.unsqueeze(-1)
                else:
                    negative = torch.cat([negative, neg.unsqueeze(-1)], 1)
                    
            
            cmi_dim = (positive.unsqueeze(-1)- negative ).mean()

            cmi_dims.append(cmi_dim.abs().item())

       
        return cmi_dims
    
    def loglikeli(self, x_samples, y_samples):
        x_samples = x_samples.clone()
        y_samples = y_samples.clone()
        
        num = y_samples.shape[1]
        max_values= torch.mean(x_samples, dim=1)
        for k in range(y_samples.shape[1]):

            
            mu, logvar = self.get_mu_logvar(max_values)
            lg = (-(mu - y_samples[:,k].unsqueeze(-1))**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
            if  k == 0:
                lgs = lg
            else:
                lgs += lg

        del x_samples, y_samples
        torch.cuda.empty_cache()
        #print("lg", lg)
        return lgs/num
    
    def loglikeli_mask(self, x_samples, y_samples):
        negative = []
        x_samples = x_samples.clone()
        y_samples = y_samples.clone()
        num = y_samples.shape[1]
        for k in range(y_samples.shape[1]):
            for i in range(x_samples.shape[1]-1):
                x_new = x_samples.clone()
                x_new[:, i, :] = float(0) #neg_inf  # 使用fill_方法替换指定维度为负无穷

                #x_new = x_samples.masked_fill(x_samples.new_zeros(x_samples.size()).bool()[:, i, :], float('-inf'))
                #max_values, max_indices = torch.max(x_new, dim=1)
                max_values = torch.mean(x_new, dim=1)

                mu, logvar = self.get_mu_logvar_neg(max_values)
                neg =  (-(mu - y_samples[:,k].unsqueeze(-1))**2 /logvar.exp()-logvar).sum(dim=-1) #(- (mu - y_samples)**2 /2./logvar.exp() - logvar/2.).sum(dim = -1) #[nsample]
                if i == 0:
                    negative = neg.unsqueeze(-1)
                else:
                    negative = torch.cat([negative, neg.unsqueeze(-1)], 1)
            if k == 0:
                negatives = negative.sum(dim=1).mean(dim=0)
            else:
                negatives += negative.sum(dim=1).mean(dim=0)
        del x_samples, y_samples
        torch.cuda.empty_cache()
        #print('mask', negative.sum(dim=1).mean(dim=0))
        return negatives/num

    def learning_loss(self, x_samples, y_samples):
        x_samples = x_samples.unsqueeze(-1)
        x_samples = self.linear_map(x_samples)

        return  - self.loglikeli_mask(x_samples, y_samples)  - self.loglikeli(x_samples, y_samples)

# %%


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


# %%
Dim = 5
dataset = CMINE.create_dataset_DGP(GenModel="", Params="", Dim=5, N=64)
s_t = torch.from_numpy(dataset[0]).float().cuda()
s_next = torch.from_numpy(dataset[1]).float().cuda()
a = torch.from_numpy(dataset[2]).float().cuda()

# %%
s_next.shape

# %%
torch.cat([s_t,a], dim=1).shape

# %%

# %%
# %%
sample_dim = 2*Dim
batch_size = 64
hidden_size = 15
learning_rate = 0.005
training_steps = 40

cubic = False 

# %%
model = L1OutUB(sample_dim + 1, sample_dim, hidden_size).cuda()
optimizer = torch.optim.Adam(model.parameters(), learning_rate)

# %%

# %%
mi_est_values = []

# %%
for step in range(training_steps):
    #batch_x, batch_y = sample_correlated_gaussian(rho, dim=sample_dim, batch_size = batch_size, to_cuda = True, cubic = cubic)
    dataset = CMINE.create_dataset_DGP(GenModel="", Params="", Dim=Dim, N=64)
    s_t = torch.from_numpy(dataset[0]).float().cuda()
    s_next = torch.from_numpy(dataset[1]).float().cuda()
    a = torch.from_numpy(dataset[2]).float().cuda()
    
    batch_x = torch.cat([s_t,a], dim=1)
    batch_y = s_next
    model.eval()
    cmi = model(batch_x, batch_y)
    mi_est_values.append(cmi)
    #print(cmi)
    # %%
    model.train() 

    model_loss = model.learning_loss(batch_x, batch_y)

    optimizer.zero_grad()
    model_loss.backward(retain_graph=True)
    optimizer.step()

    del batch_x, batch_y
    torch.cuda.empty_cache()
#print("finish training for %s with true MI value = %f"%('LOO', 6.0))
print(np.array(mi_est_values).mean())

# %%
batch_x = torch.cat([s_t,a], dim=1)
batch_x.shape

# %% [markdown]
# # Comments
# The results of:
# 
# '''
# np.array(mi_est_values).mean(axis=0)
# 
# array([3.03903936e+00, 5.72949738e+00, 2.74534021e+00, 2.70783820e+00,
#        2.98499811e+00, 2.96300891e+01, 5.01361554e+02, 4.31678332e+05,
#        3.39265994e+03, 2.90363561e+02])
# 
# '''
# 
# looks good as: S_t is combined with [Xex_t, Se_t] where Xex_t is only involed by themselves and Se_t can have infulence on next states.

# %%
np.array(mi_est_values).mean(axis=0)

# %%



