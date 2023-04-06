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

import numpy as np
from scipy import stats

import numpy as np
from sklearn.neighbors import KernelDensity

def gaussian_conditional_mutual_info_highdim(x, y, z, bandwidth=1.0, kernel='gaussian'):
    """
    Calculates the Gaussian conditional mutual information between two high-dimensional variables x and y given a third variable z.
    """
    n = len(x)
    xyz = np.column_stack((x, y, z))
    
    # Fit kernel density estimators
    kde_xz = KernelDensity(bandwidth=bandwidth, kernel=kernel).fit(xyz[:, :2])
    kde_yz = KernelDensity(bandwidth=bandwidth, kernel=kernel).fit(xyz[:, 1:])
    kde_xyz = KernelDensity(bandwidth=bandwidth, kernel=kernel).fit(xyz)
    
    # Calculate the entropy of x given z
    log_density_xz = kde_xz.score_samples(xyz[:, :2])
    entropy_x_given_z = -np.mean(log_density_xz)
    
    # Calculate the entropy of y given z
    log_density_yz = kde_yz.score_samples(xyz[:, 1:])
    entropy_y_given_z = -np.mean(log_density_yz)
    
    # Calculate the joint entropy of x and y given z
    log_density_xyz = kde_xyz.score_samples(xyz)
    entropy_xy_given_z = -np.mean(log_density_xyz)
    
    # Calculate the conditional mutual information
    mi_xy_given_z = entropy_x_given_z + entropy_y_given_z - entropy_xy_given_z
    return mi_xy_given_z
'''
def conditional_mutual_information(x, y, z):
    """
    Computes the conditional mutual information I(x,y|z) for three random variables x, y, and z.
    x, y, z: numpy arrays of shape (n,) containing the values of the random variables.
    """
    # Compute the joint probabilities
    p_xyz, _ = np.histogramdd((x, y, z), bins=10)
    p_xyz /= np.sum(p_xyz)
    
    # Compute the marginal probabilities
    p_xz = np.sum(p_xyz, axis=1)
    p_yz = np.sum(p_xyz, axis=0)
    p_z = np.sum(p_xyz)
    
    # Compute the conditional probabilities
    p_x_given_z = p_xz / p_z
    p_y_given_z = p_yz / p_z
    
    # Compute the mutual information
    mi = stats.entropy(p_xyz.flatten())
    mi -= stats.entropy(p_xz.flatten())
    mi -= stats.entropy(p_yz.flatten())
    
    # Compute the conditional mutual information
    cmi = mi - stats.entropy(p_x_given_z.flatten()) - stats.entropy(p_y_given_z.flatten())
    
    return cmi



class Data_guassian(object):
    def __init__(self, data, points=50):
        self.data = data
        self.means = self.data.mean(axis=0)
        self.cov = np.cov(self.data.T)
        self.df = pd.DataFrame(data, columns=['x1', 'x2', 'x3'])
        self.p_xyz = multivariate_normal(self.means, self.cov)
        self.p_xz = multivariate_normal(self.means[[0, 2]], self.cov[[0, 2]][:, [0, 2]])
        self.p_yz = multivariate_normal(self.means[[1, 2]], self.cov[[1, 2]][:, [1, 2]])
        self.p_z = multivariate_normal(self.means[2], self.cov[2, 2])
        self.x_vals = np.linspace(self.df.x1.min(), self.df.x1.max(), num=points, endpoint=True)
        self.y_vals = np.linspace(self.df.x2.min(), self.df.x2.max(), num=points, endpoint=True)
        self.z_vals = np.linspace(self.df.x3.min(), self.df.x3.max(), num=points, endpoint=True)

    def get_cmi(self):
        x_vals = self.x_vals
        y_vals = self.y_vals
        z_vals = self.z_vals
        prod = itertools.product(*[x_vals, y_vals, z_vals])

        p_z = self.p_z
        p_xz = self.p_xz
        p_yz = self.p_yz
        p_xyz = self.p_xyz
        quads = ((p_xyz.pdf([x, y, z]), p_z.pdf(z), p_xz.pdf([x, z]), p_yz.pdf([y, z])) for x, y, z in prod)

        cmi = sum((xyz * (np.log(z) + np.log(xyz) - np.log(xz) - np.log(yz)) for xyz, z, xz, yz in quads))
        return cmi
'''    

def estimate_CMI(config):
    #-----------------------------------------------------------------#    
    #--------------- Create the dataset ------------------------------#
    #-----------------------------------------------------------------#    
    dim = config.d
    n = config.n
    
    sigma_x = config.sigma_x
    sigma_1 = config.sigma_y
    sigma_2 = config.sigma_z
    arrng = config.arrng
    
    params = (sigma_x,sigma_1,sigma_2)
    
    if config.scenario == 0: #Estimate I(X;Y|Z)
        True_CMI = -dim*0.5*np.log(sigma_1**2 * (sigma_x**2+sigma_1**2 + sigma_2**2)/((sigma_x**2 + sigma_1**2)*(sigma_1**2 + sigma_2**2)))
    elif config.scenario == 1: #Estimate I(X;Z|Y)    
        True_CMI = 0
    
    K = config.k
    b_size = config.batch_size
    
    #----------------------------------------------------------------------#
    #------------------------Train the network-----------------------------#
    #----------------------------------------------------------------------#
    
    # Set up neural network paramters
    LR = config.lr
    EPOCH = config.e
    SEED = config.seed
    input_size = 2*dim +1 +2*dim #3*dim
    hidden_size = 64
    num_classes = 2
    tau = config.tau
    
    NN_params = (input_size,hidden_size,num_classes,tau)
    EVAL = False
    
    #Monte Carlo param
    T = config.t
    S = config.s
    
    CMI_LDR = []
    CMI_DV = []
    CMI_NWJ = []

    #kernel based method
    ker = config.ker
    # RL setting
    rl = config.rl
    
    for s in range(S):
        CMI_LDR_t = []
        CMI_DV_t = []
        CMI_NWJ_t = []
            
        #Create dataset
        #
        if rl ==1: 
            dataset = CMINE.create_dataset_DGP(GenModel="", Params="", Dim=dim, N=n)
        else:
            dataset = CMINE.create_dataset(GenModel='Gaussian_nonZero', Params=params, Dim=dim, N=n)

        if ker == 1:
            cmi_guassian = gaussian_conditional_mutual_info_highdim(dataset[0],dataset[1],dataset[2])
            print('Guassion=',cmi_guassian) 



        for t in range(T): 
            start_time = time.time()
            
            batch_train, target_train, joint_test, prod_test = CMINE.batch_construction(data=dataset, arrange=arrng, set_size=b_size, K_neighbor=K)    
            print('Duration of data preparation: ',time.time()-start_time,' seconds')
            
            CMI_LDR_Eval=[]
            CMI_DV_Eval=[]
            CMI_NWJ_Eval=[]

            start_time = time.time()
            #Train
            if EVAL:
                model, loss_e, CMI_LDR_e, CMI_DV_e, CMI_NWJ_e = CMINE.train_classifier(BatchTrain=batch_train, TargetTrain=target_train, Params=NN_params, Epoch=EPOCH, Lr=LR, Seed=SEED, Eval=True, JointEval=joint_test, ProdEval=prod_test)        
                CMI_LDR_Eval.append(CMI_LDR_e)
                CMI_DV_Eval.append(CMI_DV_e)    
                CMI_NWJ_Eval.append(CMI_NWJ_e)
            else:   
                model, loss_e = CMINE.train_classifier(BatchTrain=batch_train, TargetTrain=target_train, Params=NN_params, Epoch=EPOCH, Lr=LR, Seed=SEED)
            
            #Compute I(X;Y|Z)
            CMI_est = CMINE.estimate_CMI(model, joint_test, prod_test)
            print(CMI_est)
        
            print('Duration: ', time.time()-start_time, ' seconds')       
            
            print('LDR=',CMI_est[0])   
            print('DV=',CMI_est[1])   
            print('NWJ=',CMI_est[2]) 
            if rl != 1:
                print('True=',True_CMI)
            else:
                print('True=Todo')
            if ker == 1:
                print('Guassion=',cmi_guassian) 
            
            CMI_LDR_t.append(CMI_est[0])
            CMI_DV_t.append(CMI_est[1])
            CMI_NWJ_t.append(CMI_est[2])
            
        CMI_LDR.append(np.mean(CMI_LDR_t))
        CMI_DV.append(np.mean(CMI_DV_t))
        CMI_NWJ.append(np.mean(CMI_NWJ_t))    
        
    file = open(config.directory+'/result_'+str(config.seed), 'wb')
    pickle.dump((True_CMI,CMI_LDR,CMI_DV,CMI_NWJ,CMI_LDR_Eval,CMI_DV_Eval,CMI_NWJ_Eval,n,dim,K,LR,EPOCH,loss_e), file)
    
    file.close()    
    


def estimate_CMI_DPI(config):
    #-----------------------------------------------------------------#    
    #--------------- Create the dataset ------------------------------#
    #-----------------------------------------------------------------#    
    dim = config.d
    dim_split=config.dim_split
    n = config.n
    

    sigma_x = config.sigma_x
    sigma_1 = config.sigma_y
    sigma_2 = config.sigma_z
    
    q = config.q
    
    eye2 = np.eye(dim)
    eye2[0,1] = q
    eye2[dim-1,dim-2] = q
    for i in range(1,dim-1):
        eye2[i,i-1] = eye2[i,i+1] = q    
    Sigma_x = eye2 * sigma_x**2 
    Sigma_1 = eye2 * sigma_1**2
    Sigma_2 = eye2 * sigma_2**2
    
    params = (Sigma_x, Sigma_1, Sigma_2, dim_split)

    arrng = config.arrng
    
    K = config.k
    b_size = config.batch_size
    
    #----------------------------------------------------------------------#
    #------------------------Train the network-----------------------------#
    #----------------------------------------------------------------------#
    
    # Set up neural network paramters
    LR = config.lr
    EPOCH = config.e
    SEED = config.seed
    #input_size = 3*dim
    input_size = 2*dim + 1
    hidden_size = 64
    num_classes = 2
    tau = config.tau
    
    NN_params = (input_size,hidden_size,num_classes,tau)
    EVAL = False
    
    #Monte Carlo param
    T = config.t
    S = config.s
    
    CMI_LDR = []
    CMI_DV = []
    CMI_NWJ = []
    
    CMI_LDR_0 = []
    CMI_DV_0 = []
    CMI_NWJ_0 = []
    
    
    CMI_LDR_1 = []
    CMI_DV_1 = []
    CMI_NWJ_1 = []
    
    CMI_LDR_2 = []
    CMI_DV_2 = []
    CMI_NWJ_2 = []
    
    
    True_CMI = 0.5*np.log(det(Sigma_x+Sigma_1)/det(Sigma_1)) - 0.5*np.log(det(Sigma_x+Sigma_1+Sigma_2)/det(Sigma_1+Sigma_2))
    for s in range(S):
        ########################################################
        #   Compute I(X;Y|Z)=I(X;Y_1|Z)+ I(X;Y_2|Z,Y_1)
        #   I1=I(X;Y_1|Z)
        #   I2=I(X;Y_2|Z,Y_1)
        ########################################################
    
        for mode in range(3):
            if mode == 0:  # I(X;Y|Z)   
                arrng = [[0],[1],[2]]
                input_size = 3*dim
                NN_params = (input_size,hidden_size,num_classes,tau)    
            elif mode == 1: # I1=I(X;Y_1|Z)
                arrng = [[0],[3],[2]]
                input_size = 2*dim + dim_split
                NN_params = (input_size,hidden_size,num_classes,tau)
            elif mode == 2: # I1=I(X;Y_2|Z Y_1)
                arrng = [[0],[4],[2,3]]
                input_size = 3*dim
                NN_params = (input_size,hidden_size,num_classes,tau)
            
            CMI_LDR_t = []
            CMI_DV_t = []
            CMI_NWJ_t = []
                
            #Create dataset
            dataset = CMINE.create_dataset(GenModel='Gaussian_Correlated_split',Params=params, Dim=dim, N=n)
            #dataset=CMINE.create_dataset(GenModel='Gaussian_Split',Params=params, Dim=dim, N=n)
                
            for t in range(T): 
                
                print('s,t= ',s,t)
                start_time = time.time()
                
                batch_train, target_train, joint_test, prod_test=CMINE.batch_construction(data=dataset, arrange=arrng, set_size=b_size, K_neighbor=K)    
                print('Duration of data preparation: ',time.time()-start_time,' seconds')
        
                CMI_LDR_Eval=[]
                CMI_DV_Eval=[]
                CMI_NWJ_Eval=[]
                         
                start_time = time.time()
                         
                #Train
                if EVAL:
                    model, loss_e, CMI_LDR_e, CMI_DV_e, CMI_NWJ_e = CMINE.train_classifier(BatchTrain=batch_train, TargetTrain=target_train, Params=NN_params, Epoch=EPOCH, Lr=LR, Seed=SEED, Eval=True, JointEval=joint_test, ProdEval=prod_test)        
                    CMI_LDR_Eval.append(CMI_LDR_e)
                    CMI_DV_Eval.append(CMI_DV_e)    
                    CMI_NWJ_Eval.append(CMI_NWJ_e)
                else:   
                    model, loss_e = CMINE.train_classifier(BatchTrain=batch_train, TargetTrain=target_train, Params=NN_params, Epoch=EPOCH, Lr=LR, Seed=SEED)
                
                #Compute I(X;Y|Z)
                CMI_est = CMINE.estimate_CMI(model, joint_test, prod_test)
            
                print('Duration: ', time.time()-start_time, ' seconds')       
                print('Mode= ',mode) 
                print('DV=',CMI_est[1])   
                print('True=',True_CMI)
                
                CMI_LDR_t.append(CMI_est[0])
                CMI_DV_t.append(CMI_est[1])
                CMI_NWJ_t.append(CMI_est[2])
            
            if mode==0:
                CMI_LDR_0.append(np.mean(CMI_LDR_t))
                CMI_DV_0.append(np.mean(CMI_DV_t))
                CMI_NWJ_0.append(np.mean(CMI_NWJ_t))  
            elif mode==1:
                CMI_LDR_1.append(np.mean(CMI_LDR_t))
                CMI_DV_1.append(np.mean(CMI_DV_t))
                CMI_NWJ_1.append(np.mean(CMI_NWJ_t))  
            elif mode==2:
                CMI_LDR_2.append(np.mean(CMI_LDR_t))
                CMI_DV_2.append(np.mean(CMI_DV_t))
                CMI_NWJ_2.append(np.mean(CMI_NWJ_t))  
        
        CMI_LDR.append([CMI_LDR_0[-1],CMI_LDR_1[-1],CMI_LDR_2[-1]])
        CMI_DV.append([CMI_DV_0[-1],CMI_DV_1[-1],CMI_DV_2[-1]])
        CMI_NWJ.append([CMI_NWJ_0[-1],CMI_NWJ_1[-1],CMI_NWJ_2[-1]])
        
        print('DV:   ',CMI_DV_0[-1],' - ',CMI_DV_1[-1],' - ',CMI_DV_2[-1])
        print('True=',True_CMI)
        print('\n\n')   
        
    file = open(config.directory+'/result_'+str(config.seed), 'wb')
    pickle.dump((True_CMI,CMI_LDR,CMI_DV,CMI_NWJ,CMI_LDR_Eval,CMI_DV_Eval,CMI_NWJ_Eval,n,dim,K,LR,EPOCH,loss_e), file)
    
    file.close()        


