import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import itertools

np.random.seed(37)

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


def get_serial(N=1000):
    x1 = np.random.normal(1, 1, N)
    x3 = np.random.normal(1 + 3.5 * x1, 1, N)
    x2 = np.random.normal(1 - 2.8 * x3, 3, N)

    data = np.vstack([x1, x2, x3]).T
    means = data.mean(axis=0)
    cov = np.cov(data.T)

    return Data(data, means, cov)

def get_diverging(N=1000):
    x3 = np.random.normal(1, 1, N)
    x1 = np.random.normal(1 + 2.8 * x3, 1, N)
    x2 = np.random.normal(1 - 2.8 * x3, 3, N)

    data = np.vstack([x1, x2, x3]).T
    means = data.mean(axis=0)
    cov = np.cov(data.T)

    return Data(data, means, cov)

def get_converging(N=1000):
    x1 = np.random.normal(2.8, 1, N)
    x2 = np.random.normal(8.8, 3, N)
    x3 = np.random.normal(1 + 0.8 * x1 + 0.9 * x2, 1, N)


    data = np.vstack([x1, x2, x3]).T
    means = data.mean(axis=0)
    cov = np.cov(data.T)

    return Data(data, means, cov)



if __name__ == "__main__":
    m_s = get_serial()
    m_d = get_diverging()
    m_c = get_converging()