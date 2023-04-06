
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

def exo_endo_mdp_linear(MDP_info, T, policy):
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
        [X_, Endo_, Rx_, Re_, R, A_] = env(MDP_info, T, policy)
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

