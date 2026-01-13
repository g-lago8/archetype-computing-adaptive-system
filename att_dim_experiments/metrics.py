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



def compute_corr_dim(trajectory: np.ndarray, k1=5,  k2=20, transient=4000) -> list[float]:
    """Compute Correlation Dimension for each module in a set of trajectories

    Args:
        trajectory (np.ndarray): trajectory of shape (N_steps, N_modules, N_h)
        k1 (int, optional): k1 :  First neighborhood size considered. Defaults to 5.
        k2 (int, optional): k2 :  Second neighborhood size considered. Defaults to 20.
        transient (int, optional): transient :  Number of initial steps to discard. Defaults to 4000.

    Returns:
        Optional[list[float]]: List of correlation dimension estimates for each module, or None if an error occurs
    """
    corr_dim_values = []
    for i in range(trajectory.shape[1]): # for each module in the network
        corr_dim_estimator = CorrInt(k1=k1, k2=k2)
        traj_i = trajectory[transient:, i]
        corr_dim = corr_dim_estimator.fit_transform(traj_i)
        corr_dim_values.append(corr_dim)

    return corr_dim_values


def compute_participation_ratio(trajectory, transient=4000) -> list[float]:
    """
    Compute Participation Ratio for each module in a set of trajectories
    Args:
        trajectory (np.ndarray): trajectory of shape (N_steps, N_modules, N_h)
        transient (int, optional): transient :  Number of initial steps to discard. Defaults to 4000.
    Returns:
        Optional[list[float]]: List of participation ratio estimates for each module, or None if an error occurs
    """
    n_modules = trajectory.shape[1]
    participation_ratios = []
    for i in range(n_modules):
        traj_i = trajectory[transient:, i]
        # compute covariance matrix
        cov_matrix = np.cov(traj_i, rowvar=False)
        # compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        # compute participation ratio
        pr = (np.sum(eigenvalues))**2 / np.sum(eigenvalues**2)
        participation_ratios.append(pr)
    return participation_ratios


def compute_effective_rank(trajectory, transient=4000, eps = 1e-10) -> list[float]:
    """
    Compute Effective Rank for each module in a set of trajectories
    Args:
        trajectory (np.ndarray): trajectory of shape (N_steps, N_modules, N_h)
        transient (int, optional): transient :  Number of initial steps to discard. Defaults to 4000.
    Returns:
        Optional[list[float]]: List of effective rank estimates for each module, or None if an error occurs
    """
    n_modules = trajectory.shape[1]
    ranks = []
    for i in range(n_modules):
        traj_i = trajectory[transient:, i]
        singvals = np.linalg.svdvals(traj_i)
        s = np.sum(np.abs(singvals))
        n_singvals = singvals / s
        entropy = - np.dot(n_singvals + eps, np.log(n_singvals + eps)) 
        ranks.append(np.exp(entropy))
    return ranks


def nrmse(preds: np.ndarray, target: np.ndarray) -> float:
    mse = np.mean(np.square(preds - target))
    norm = np.sqrt(np.mean(np.square(target)))
    # rmse / norm
    return np.sqrt(mse) / (norm + 1e-9)



__all__ = ['compute_corr_dim', 'compute_participation_ratio', 'compute_effective_rank', 'nrmse']

if __name__ == '__main__':
    # simple test
    traj = np.random.rand(5000, 1, 2)
    print("Correlation Dimension:", compute_corr_dim(traj))
    print("Participation Ratio:", compute_participation_ratio(traj))
    print("Effective Rank:", compute_effective_rank(traj))