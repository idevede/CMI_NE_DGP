# %%
import time
import numpy as np
import pickle
from numpy.linalg import det

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

class DR_CMI(nn.Module):  # naive upper bound
    def __init__(self, x_dim, y_dim, hidden_size):
        super(DR_CMI, self).__init__()
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
        #print("x_samples",x_samples[:, -1])
        propensity_score = torch.sigmoid(propensity)
        # y = ( x_samples[:,-1] == -5).float() ## 将类别标签转换为 0 和 1
        # loss = F.binary_cross_entropy(out, y.unsqueeze(-1))
        propensity_score = torch.where(propensity_score < 0.0001, torch.tensor([0.0001]).to(propensity_score.device), propensity_score)
        propensity_score = torch.where(propensity_score > 0.9999, torch.tensor([0.9999]).to(propensity_score.device), propensity_score)
        #print(propensity_score)
        w_1 = 1 / (propensity_score)
        w_0 = 1 / (1-propensity_score)
        # print(w_1)
        # print(w_0)
        
        inds_1 = np.where(x_samples.cpu().numpy()[:,-1] ==1.0 ) # 找到B中不为0的位置
        inds_0 = np.where(x_samples.cpu().numpy()[:,-1] ==0.0 ) # 找到B中不为0的位置
        
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

            dr_0 = w_0[inds_0]*((positive.unsqueeze(-1)- negative)[inds_0] -cmi_dim_0)
            dr_1 = w_1[inds_1]*((positive.unsqueeze(-1)- negative)[inds_1] -cmi_dim_1)
            if  torch.isnan(dr_0.mean()) or torch.isnan(dr_1.mean()):
                dr = (positive.unsqueeze(-1)- negative).mean()#0.5*cmi_dim_0  + 0.5*cmi_dim_1
                #print(positive.unsqueeze(-1)- negative)
                # print(cmi_dim_0)
                # print(dr)
                #0.5*((cmi_dim_0 + w_0[inds_0]*((positive.unsqueeze(-1)- negative)[inds_0] -cmi_dim_0))).mean()
            else:
                dr = 0.5*((cmi_dim_0 + dr_0)).mean() + 0.5*((cmi_dim_1 + dr_1)).mean()
            # print((w_0[inds_0]*((positive.unsqueeze(-1)- negative)[inds_0] -cmi_dim_0)).mean())
            # print((w_1[inds_1]*((positive.unsqueeze(-1)- negative)[inds_1] -cmi_dim_1)).mean())
            # if not torch.isnan((w_1[inds_1]*((positive.unsqueeze(-1)- negative)[inds_1] -cmi_dim_1)).mean() ):
            #     dr += 0.5*((cmi_dim_1 + w_1[inds_1]*((positive.unsqueeze(-1)- negative)[inds_1] -cmi_dim_1))).mean()
            # else:
            #     dr += 0.5*cmi_dim_1
           
            #cmi_dims.append((cmi_dim_0+cmi_dim_1).abs().item()/2)
            
            drs.append(dr.abs().item())

        #print(drs)
        return drs
    
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
        out = torch.sigmoid(propensity)
        #y = ( x_samples[:,-1] == -5).float() ## 将类别标签转换为 0 和 1
        y = ( x_samples[:,-1] == 1.0).float() ## 将类别标签转换为 0 和 1
        #print(y)
        loss = F.binary_cross_entropy(out, y.unsqueeze(-1))

        x_samples = x_samples.unsqueeze(-1)
        x_samples = self.linear_map(x_samples)

        return  - self.loglikeli_mask(x_samples, y_samples)  - self.loglikeli(x_samples, y_samples) +  loss

    
    
class CDL_CMI(nn.Module):  # naive upper bound
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CDL_CMI, self).__init__()
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
                    
            
            cmi_dim = (positive.unsqueeze(-1)- negative ).mean() # 64,10 ->1
            cmi_dims.append(cmi_dim.abs().item())    

        #print(drs)
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
        
        x_samples = x_samples.unsqueeze(-1)
        x_samples = self.linear_map(x_samples)

        return  - self.loglikeli_mask(x_samples, y_samples)  - self.loglikeli(x_samples, y_samples) 