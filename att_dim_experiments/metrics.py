"""
Code to compute the attractor dimension metrics in the paper ---
List of metrics considered:
- Correlation dimension (CD)
- Kernel rank (KR)
- Participation ratio (PR)
- Maximum Lyapunov exponent (MLE)
"""


import numpy as np
from typing import Optional
from skdim.id import CorrInt



def compute_corr_dim(trajectory: np.ndarray, k1=5,  k2=20, transient=4000) -> Optional[list[float]]:
    corr_dim_values = []
    for i in range(trajectory.shape[1]): # for each module in the network
        corr_dim_estimator = CorrInt(k1=k1, k2=k2)
        try:
            traj_i = trajectory[transient:, i]
            corr_dim = corr_dim_estimator.fit_transform(traj_i)
            corr_dim_values.append(corr_dim)
        except Exception as e:
            print(f"Error computing correlation dimension: {e}")
            return None
    return corr_dim_values


def compute_participation_ratio(trajectory, transient=4000):
    n_modules = trajectory.shape[1]
    participation_ratios = []
    for i in range(n_modules):
        try:
            traj_i = trajectory[transient:, i]
            # compute covariance matrix
            cov_matrix = np.cov(traj_i, rowvar=False)
            # compute eigenvalues
            eigenvalues = np.linalg.eigvalsh(cov_matrix)
            # compute participation ratio
            pr = (np.sum(eigenvalues))**2 / np.sum(eigenvalues**2)
            participation_ratios.append(pr)
        except Exception as e:
            print(e)
            return None
    return participation_ratios


def compute_effective_rank(trajectory, transient=4000, eps = 1e-10):
    n_modules = trajectory.shape[1]
    ranks = []
    for i in range(n_modules):
        traj_i = trajectory[transient:, i]
        try:
            singvals = np.linalg.svdvals(traj_i)
            s = np.sum(np.abs(singvals))
            n_singvals = singvals / s
            entropy = - np.dot(n_singvals + eps, np.log(n_singvals + eps)) 
            ranks.append(np.exp(entropy))
        except Exception as e:
            print(e)
            return None
    return ranks


if __name__ == '__main__':
    # simple test
    traj = np.random.rand(5000, 1, 2)
    print("Correlation Dimension:", compute_corr_dim(traj))
    print("Participation Ratio:", compute_participation_ratio(traj))
    print("Effective Rank:", compute_effective_rank(traj))