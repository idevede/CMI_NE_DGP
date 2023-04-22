
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed
from copy import deepcopy
import pickle

class config():
  # metadata = {'render.modes': ['human']}

  def __init__(self, horizon, p, gamma = 1, nA = 2, seed =1):

    self.horizon = horizon
    self.p = p # state dimension
    self.nA = nA
    self.gamma = gamma
    return

def setminus(a,x):
    return [y for y in a if y != x]

def normalize_M(e):
    return (e.T/(1.01*e.sum(axis=1)) ).T

def gen_M(p_endo, p_exo, p_A):
    Mx = np.random.normal(size=(p_exo, p_exo), loc=0.2, scale=1)
    Mendo = np.random.normal(size=(p_endo,p_exo+p_endo+p_A), loc=-0.2, scale=1)
    Mx = normalize_M(Mx); Mendo = normalize_M(Mendo)
    return [Mx,Mendo]

def exo_endo_mdp_linear(MDP_info, T, policy):  ## This is env
    [Mx, Mendo, p_endo, p_exo, p_A] = MDP_info
    X__ = np.zeros([T,p_exo]); Endo__ = np.zeros([T,p_endo]); Rx__=np.zeros(T); Re__=np.zeros(T)
    A__ = np.zeros([T,p_A])
    Endo_t = np.zeros(p_endo); X_t = np.zeros(p_exo)
    for t in range(T):
        A_t = policy(Endo_t,X_t)
        X_t1 = Mx @ X_t + np.random.normal(size=p_exo, loc=0, scale=0.09)
        dummy = np.zeros(X_t.shape)
        Endo_t1 = (Mendo @ np.expand_dims(np.hstack([Endo_t, dummy, A_t]), axis=1)).flatten() + np.random.normal(size=p_endo, loc=0, scale=0.04)
        Rxt = (-3*np.mean(X_t))+ np.random.normal(loc=0, scale=0.09)
        Ret = np.mean(Endo_t) + np.mean(A_t * Endo_t) #np.exp(-np.log(np.abs(np.mean(Endo_t) - 1)) )+ np.random.normal(loc=0, scale=0.03)
        X__[t,:] = X_t1; Endo__[t,:] = Endo_t1; Rx__[t] = Rxt; Re__[t] = Ret
        X_t = X_t1
        Endo_t = Endo_t1
        A__[t] = A_t

    R=Rx__+Re__
    return [X__, Endo__, Rx__, Re__, R, A__]

def get_trajectories(N_traj, env, MDP_info, T, policy):
    [Mx, Mendo, p_endo, p_exo, p_A] = MDP_info
    X__ = np.zeros([N_traj, T,p_exo]); Endo__ = np.zeros([N_traj, T,p_endo]);
    Rx__=np.zeros([N_traj,T]); Re__=np.zeros([N_traj,T])
    A__ = np.zeros([N_traj, T,p_A])
    for n in range(N_traj):
        [X_, Endo_, Rx_, Re_, R, A_] = env(MDP_info, T, policy) # exo_endo_mdp_linear(MDP_info, T, policy)
        X__[n,:]=X_; Endo__[n,:] = Endo_; Rx__[n,:] = Rx_; Re__[n,:] = Re_; A__[n,:] = A_
    return [X__, Endo__, Rx__, Re__, A__]

def fitted_Q_evaluation(config_, regressor, policy, S_,A_,Y_,R_):
    nA = config_.nA
    [N, horizon, p] = S_.shape

    Qsx = dict()

    for t in reversed(range(horizon)):
        policy_a=[None]*nA
        # for a in range(nA):
        Qsx[t] = regressor()
        S_t = slice_S(S_,t)
        data = np.hstack([ S_t, A_[:,t].reshape([N,1]) ])
        if t < horizon-1:
            data_next = [None]*nA
            S_t1=S_[:,t+1,:]
            for a in range(nA):
                policy_a[a] = policy(t+1, S_t1, a*np.ones(N).astype(int))
                data_next[a] = np.hstack([S_t1, a*np.ones([N,1])])
                # E_\pi [Q_t+1 (S_t+1, X_t+1, a) ]
                Q_tplus1_pi = sum([ Qsx[t+1].predict(data_next[a])*policy_a[a] for a in range(nA)  ]).flatten()
                Qsx[t].fit(data, R_[:,t] + Q_tplus1_pi )
        # generate Q predictions from previous
        # fit Q
        else:
            Qsx[t].fit(data, R_[:,t])
        return Qsx

    
def policy(X,E): 
    return 5*(np.mean(E)-1)

def policy_binary(X,E): 
    return float(np.random.randint(0, 2))

def policy_conf_binary(X,E):
    prob = np.mean(X)+np.sign(np.mean(E)-1)*np.mean(E)
    prob = np.abs(prob)*np.random.normal()
    prob = np.clip(prob, 0.2, 0.8)
    #print(prob)
    return float(np.random.choice([0, 1], size=1, p=[prob , 1-prob]))

def noisy_policy(X,E): 
    return np.mean(X)+np.sign(np.mean(E)-1)*np.mean(E) + np.random.normal()



def create_dataset_DGP(Dim, N, T = 5):
    # If the dataset=(x,y,z) we compute I(X;Y|Z) 
    p_endo= Dim
    p_exo = Dim
    p = p_endo+p_exo
    p_A = 1 # the dimension of the action space
    [Mx,Mendo] = gen_M(p_endo, p_exo, p_A)
    MDP_info = [Mx,Mendo,p_endo,p_exo,p_A]   
    
    #T = T
    # [X_, Endo_, Rx_, Re_, R, A_] = exo_endo_mdp_linear(MDP_info, T, policy)
    N_traj = N
    [X__, Endo__, Rx__, Re__, A__] = get_trajectories(N_traj, exo_endo_mdp_linear, MDP_info, T, policy)
    # (S_t,A_t, R_t,S_{t+1}) tuples for t = 1,T-1
    StA = np.concatenate([X__, Endo__, A__],axis = 2)[:,:-1,:]
    StA_next = np.concatenate([X__, Endo__],axis=2)[:,1:,:] # start from t=1
    # column indices: 
    # 0:p is S_t, p:p+p_a is action variable, p+p_a:-2 is S_t+1, -1 is reward'
    # 0:pexo is exogenous; pexo:-1 is endo 
    # N_traj x T-1 x (p + p_A + 1 + p) (state, action, reward, s_t1)
    StA_St1A = np.concatenate([StA,StA_next, np.expand_dims(Re__[:,:-1],2)],axis=2) ## S_{t}[0:30] A[30] S_{t+1}[31:61] R[61]
    
    dataset = [StA[:,0,:-1],StA_next[:,0,:],StA[:,0,-1:]]


    return dataset

def create_dataset_DGP_binary_A(Dim, N, T = 5):
    # If the dataset=(x,y,z) we compute I(X;Y|Z) 
    p_endo= Dim
    p_exo = Dim
    p = p_endo+p_exo
    p_A = 1 # the dimension of the action space
    [Mx,Mendo] = gen_M(p_endo, p_exo, p_A)
    MDP_info = [Mx,Mendo,p_endo,p_exo,p_A]   
    
    #T = T
    # [X_, Endo_, Rx_, Re_, R, A_] = exo_endo_mdp_linear(MDP_info, T, policy)
    N_traj = N
    [X__, Endo__, Rx__, Re__, A__] = get_trajectories(N_traj, exo_endo_mdp_linear, MDP_info, T, policy_binary)
    # (S_t,A_t, R_t,S_{t+1}) tuples for t = 1,T-1
    StA = np.concatenate([X__, Endo__, A__],axis = 2)[:,:-1,:]
    StA_next = np.concatenate([X__, Endo__],axis=2)[:,1:,:] # start from t=1
    # column indices: 
    # 0:p is S_t, p:p+p_a is action variable, p+p_a:-2 is S_t+1, -1 is reward'
    # 0:pexo is exogenous; pexo:-1 is endo 
    # N_traj x T-1 x (p + p_A + 1 + p) (state, action, reward, s_t1)
    StA_St1A = np.concatenate([StA,StA_next, np.expand_dims(Re__[:,:-1],2)],axis=2) ## S_{t}[0:30] A[30] S_{t+1}[31:61] R[61]
    
    dataset = [StA[:,1,:-1],StA_next[:,1,:],StA[:,1,-1:]]


    return dataset


def create_dataset_DGP_binary_A_conf(Dim, N, T = 5):
    # If the dataset=(x,y,z) we compute I(X;Y|Z) 
    p_endo= Dim
    p_exo = Dim
    p = p_endo+p_exo
    p_A = 1 # the dimension of the action space
    [Mx,Mendo] = gen_M(p_endo, p_exo, p_A)
    MDP_info = [Mx,Mendo,p_endo,p_exo,p_A]   
    
    #T = T
    # [X_, Endo_, Rx_, Re_, R, A_] = exo_endo_mdp_linear(MDP_info, T, policy)
    N_traj = N
    [X__, Endo__, Rx__, Re__, A__] = get_trajectories(N_traj, exo_endo_mdp_linear, MDP_info, T, policy_conf_binary)
    # (S_t,A_t, R_t,S_{t+1}) tuples for t = 1,T-1
    StA = np.concatenate([X__, Endo__, A__],axis = 2)[:,:-1,:]
    StA_next = np.concatenate([X__, Endo__],axis=2)[:,1:,:] # start from t=1
    # column indices: 
    # 0:p is S_t, p:p+p_a is action variable, p+p_a:-2 is S_t+1, -1 is reward'
    # 0:pexo is exogenous; pexo:-1 is endo 
    # N_traj x T-1 x (p + p_A + 1 + p) (state, action, reward, s_t1)
    StA_St1A = np.concatenate([StA,StA_next, np.expand_dims(Re__[:,:-1],2)],axis=2) ## S_{t}[0:30] A[30] S_{t+1}[31:61] R[61]
    
    dataset = [StA[:,1,:-1],StA_next[:,1,:],StA[:,1,-1:]]


    return dataset


if __name__ == '__main__':
    create_dataset_DGP_binary_A_conf( Dim=5, N=64)