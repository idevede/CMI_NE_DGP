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
import torch.nn.functional as F

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


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




torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

class DR_CML(nn.Module):  # naive upper bound
    def __init__(self, x_dim, y_dim, hidden_size):
        super(DR_CML, self).__init__()
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

        self.p_score = torch.nn.Linear(x_dim-1, 1)


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
        drs = []

        propensity = self.p_score(x_samples[:, :-1])#.squeeze(-1)
        propensity_score = F.sigmoid(propensity)
        # y = ( x_samples[:,-1] == -5).float() ## 将类别标签转换为 0 和 1
        # loss = F.binary_cross_entropy(out, y.unsqueeze(-1))
        w_1 = 1 / (propensity_score+0.0001)
        w_0 = 1 / (1-propensity_score+0.0001)
        

        # #将 mask 对应位置上的值设置为0
        # mask = torch.zeros_like(w)
        # mask[x_samples[:,-1] != -5] = 1
        # w = w.masked_fill(mask.bool(), 0)
        # mask_0 = torch.zeros_like(w)
        # mask_0[x_samples[:,-1] ==0.0 ] = 1
        # mask_1 = torch.zeros_like(w)
        # mask_1[x_samples[:,-1] ==1.0 ] = 1
        inds_1 = np.where(x_samples.cpu().numpy()[:,-1] ==1.0 ) # 找到B中不为0的位置
        inds_0 = np.where(x_samples.cpu().numpy()[:,-1] ==0.0 ) # 找到B中不为0的位置
        #result = A[inds] # 用 inds 选择 A 中的相应元素
        #print(inds_1)

        x_samples = x_samples.unsqueeze(-1)
        x_samples = self.linear_map(x_samples)

       
        neg_inf = torch.finfo(x_samples.dtype).min  


        max_values= torch.mean(x_samples, dim = 1)
        for k in range(y_samples.shape[1]):
            #max_values, max_indices = torch.max(x_samples, dim=1)
            
            mu, logvar = self.get_mu_logvar(max_values)
            

            positive = (- (mu - y_samples[:,k].unsqueeze(-1))**2 /2./logvar.exp() - logvar/2.).sum(dim = -1) #[nsample]

            negative = []
            
            for i in range(x_samples.shape[1]-1):
                x_new = x_samples.clone()
                x_new[:, i, :] = float(0) #neg_inf  
                max_values = torch.mean(x_new, dim=1)
                mu, logvar = self.get_mu_logvar_neg(max_values)
                neg = (- (mu - y_samples[:,k].unsqueeze(-1))**2 /2./logvar.exp() - logvar/2.).sum(dim = -1) #[nsample]
                if i == 0:
                    negative = neg.unsqueeze(-1)
                else:
                    negative = torch.cat([negative, neg.unsqueeze(-1)], 1)
                    
            
            cmi_dim_0 = (positive.unsqueeze(-1)- negative )[inds_0].mean() # 64,10 ->1
            cmi_dim_1 = (positive.unsqueeze(-1)- negative )[inds_1].mean() # 64,10 ->1

            dr = 0.5*((cmi_dim_0 + w_0[inds_0]*((positive.unsqueeze(-1)- negative)[inds_0] -cmi_dim_0))).mean()

            dr += 0.5*((cmi_dim_1 + w_1[inds_1]*((positive.unsqueeze(-1)- negative)[inds_1] -cmi_dim_1))).mean()
            
            # print(w_0[inds_0].mean(), w_1[inds_1].mean())
            # dr = 0.5*((cmi_dim_0 + 2*((positive.unsqueeze(-1)- negative)[inds_0] -cmi_dim_0))).mean()

            # dr += 0.5*((cmi_dim_1 + 2*((positive.unsqueeze(-1)- negative)[inds_1] -cmi_dim_1))).mean()


            cmi_dims.append((cmi_dim_0+cmi_dim_1).abs().item()/2)
            
            drs.append(dr.abs().item())

        #print(drs)
        return cmi_dims, drs
    
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
                x_new[:, i, :] = float(0) #neg_inf  # creat_mask

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
        propensity = self.p_score(x_samples[:, :-1])#.squeeze(-1)
        out = F.sigmoid(propensity)
        #y = ( x_samples[:,-1] == -5).float() ## 将类别标签转换为 0 和 1
        y = ( x_samples[:,-1] == 1.0).float() ## 将类别标签转换为 0 和 1
        #print(y)
        loss = F.binary_cross_entropy(out, y.unsqueeze(-1))

        x_samples = x_samples.unsqueeze(-1)
        x_samples = self.linear_map(x_samples)

        return  - self.loglikeli_mask(x_samples, y_samples)  - self.loglikeli(x_samples, y_samples) +  loss


Dim = 5
batch_size = 64
#dataset = CMINE.create_dataset_DGP( Dim=5, N=batch_size)
dataset = CMINE.create_dataset_DGP_binary_A( Dim=5, N=batch_size)
s_t = torch.from_numpy(dataset[0]).float().cuda()
s_next = torch.from_numpy(dataset[1]).float().cuda()
a = torch.from_numpy(dataset[2]).float().cuda()

# %%
sample_dim = 2*Dim

hidden_size = 15
learning_rate = 0.005
training_steps = 10

cubic = False 

# %%
model = DR_CML(sample_dim + 1, sample_dim, hidden_size).cuda()
optimizer = torch.optim.Adam(model.parameters(), learning_rate)

# %%

# %%
mi_est_values = []
dr_est_cm_values = []

# %%
for step in range(training_steps):
    #batch_x, batch_y = sample_correlated_gaussian(rho, dim=sample_dim, batch_size = batch_size, to_cuda = True, cubic = cubic)
    dataset = CMINE.create_dataset_DGP_binary_A(Dim=Dim, N=64)
    s_t = torch.from_numpy(dataset[0]).float().cuda()
    s_next = torch.from_numpy(dataset[1]).float().cuda()
    a = torch.from_numpy(dataset[2]).float().cuda()
    
    batch_x = torch.cat([s_t,a], dim=1)
    batch_y = s_next
    model.eval()
    cmi, drs = model(batch_x, batch_y)
    mi_est_values.append(cmi)
    dr_est_cm_values.append(drs)
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
print(np.array(dr_est_cm_values).mean())
print(np.array(mi_est_values).mean(axis=0))
print(np.array(dr_est_cm_values).mean(axis=0))