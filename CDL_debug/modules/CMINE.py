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



def estimate_CMI_dim(dim, n, k=5, b_size=64, LR= 0.001, mode=0, epoch = 100, seed =123, tau = 1e-4, t=1, s=2, arrng="", dataset="dataset" ):
    #-----------------------------------------------------------------#    
    #--------------- Create the dataset ------------------------------#
    #-----------------------------------------------------------------#    
    dim = dim
    n = n
    K = k
    b_size = b_size
    
    #----------------------------------------------------------------------#
    #------------------------Train the network-----------------------------#
    #----------------------------------------------------------------------#
    
    # Set up neural network paramters
    LR = LR
    EPOCH = epoch
    SEED = 123
    input_size = 2*dim +1 + 1 #3*dim
    hidden_size = 64
    num_classes = 2
    tau = tau
    
    NN_params = (input_size,hidden_size,num_classes,tau)
    EVAL = False
    
    #Monte Carlo param
    T = t
    S = s
    
    CMI_LDR = []
    CMI_DV = []
    CMI_NWJ = []

    #kernel based method
    ker = 0
    # RL setting
    rl = 1
    
    for s in range(S):
        CMI_LDR_t = []
        CMI_DV_t = []
        CMI_NWJ_t = []
            
        # #Create dataset
        # #
        # if rl ==1: 
        #     dataset = CMINE.create_dataset_DGP(GenModel="", Params="", Dim=dim, N=n)
        # else:
        #     dataset = CMINE.create_dataset(GenModel='Gaussian_nonZero', Params=params, Dim=dim, N=n)

        # if ker == 1:
        #     cmi_guassian = gaussian_conditional_mutual_info_highdim(dataset[0],dataset[1],dataset[2])
        #     print('Guassion=',cmi_guassian) 



        for t in range(T): 
            start_time = time.time()
            CMI_LDR_Eval=[]
            CMI_DV_Eval=[]
            CMI_NWJ_Eval=[]
            LDRs = []
            DVs = []
            NWJs = []
            print('Duration of data preparation: ',time.time()-start_time,' seconds')
            for i in range(dataset[1].shape[1]):

                CMI_LDR_es=[]
                CMI_DV_es=[]
                CMI_NWJ_es=[]

                new_dataset=[dataset[0],dataset[1][:,i].reshape(-1,1), dataset[2]]
                #print( dataset[2])
            
                batch_train, target_train, joint_test, prod_test = CMINE.batch_construction(data=new_dataset, arrange=arrng, set_size=b_size, K_neighbor=K)    
                
                
                

                start_time = time.time()
                #Train
                if EVAL:
                    model, loss_e, CMI_LDR_e, CMI_DV_e, CMI_NWJ_e = CMINE.train_classifier(BatchTrain=batch_train, TargetTrain=target_train, Params=NN_params, Epoch=EPOCH, Lr=LR, Seed=SEED, Eval=True, JointEval=joint_test, ProdEval=prod_test)        
                    CMI_LDR_es.append(CMI_LDR_e)
                    CMI_DV_es.append(CMI_DV_e)    
                    CMI_NWJ_es.append(CMI_NWJ_e)
                    if i == dataset[1].shape[1]-1:
                        CMI_LDR_Eval.append(CMI_LDR_es)
                        CMI_DV_Eval.append(CMI_DV_es)
                        CMI_NWJ_Eval.append(CMI_NWJ_es)
                else:   
                    model, loss_e = CMINE.train_classifier(BatchTrain=batch_train, TargetTrain=target_train, Params=NN_params, Epoch=EPOCH, Lr=LR, Seed=SEED)
                
                #Compute I(X;Y|Z)
                CMI_est = CMINE.estimate_CMI(model, joint_test, prod_test)
                #print(CMI_est)
            
                    
                
                
                # if rl != 1:
                #     print('True=',True_CMI)
                # else:
                #     print('True=Todo')
                
                
                LDRs.append(CMI_est[0])
                DVs.append(CMI_est[1])
                NWJs.append(CMI_est[2])

                if i == dataset[1].shape[1]-1:
                    CMI_LDR_t.append(LDRs)
                    CMI_DV_t.append(DVs)
                    CMI_NWJ_t.append(NWJs)
                    print('Print')
                    print('LDR=',LDRs)   
                    print('DV=',DVs)   
                    print('NWJ=',NWJs) 
                    print('True=Todo')
                    print('Duration: ', time.time()-start_time, ' seconds')   
                    
            
        CMI_LDR.append(np.mean(CMI_LDR_t))
        CMI_DV.append(np.mean(CMI_DV_t))
        CMI_NWJ.append(np.mean(CMI_NWJ_t))   

    return  CMI_LDR, CMI_DV, CMI_NWJ
        
    # file = open(config.directory+'/result_dim'+str(config.seed), 'wb')
    # # pickle.dump((True_CMI,CMI_LDR,CMI_DV,CMI_NWJ,CMI_LDR_Eval,CMI_DV_Eval,CMI_NWJ_Eval,n,dim,K,LR,EPOCH,loss_e), file)
    # pickle.dump((CMI_LDR,CMI_DV,CMI_NWJ,CMI_LDR_Eval,CMI_DV_Eval,CMI_NWJ_Eval,n,dim,K,LR,EPOCH,loss_e), file)
    
    # file.close()    

# LDR= [0.5308622251622674, 1.5356716355325466, 2.4325707580815985, 1.1077695434643704, 0.7977417743951639, 1.1617022157212529, 0.5684866754018341, 1.3048818493449013, 0.6651785006674207, 1.5753285269020125]
# DV= [-0.13116032257199872, -0.9696321829209105, 0.10308336125040629, -1.9104185451144506, -0.7596957159967769, -0.6549558833992304, -0.7631164965146944, -1.1588763993857463, -1.1411824910937058, -2.208163839425402]
# NWJ= [-0.4078472790881549, -9.71160771455036, -6.840103619658772, -18.34642735167583, -2.948900556535681, -3.9895649212079793, -2.21862323628229, -9.444002049316843, -4.4230733724188935, -41.39400239262123]
    