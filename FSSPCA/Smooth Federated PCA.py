"""
Early prototype of Federated Sparse PCA (FSPCA).

This is the original standalone Jupyter Notebook-based implementation,
kept for reference. The production version is in the root-level main.py.

All operators, smooth functions, and the ADMM loop are self-contained here.

Reference:
    Ciou et al., "Federated Learning for Sparse Principal Component Analysis",
    IEEE BigData 2023. arXiv:2311.08677
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import os
from sklearn import preprocessing
from scipy.sparse.linalg import svds


# ============================================================================
# Riemannian operators on the Stiefel manifold
# ============================================================================

def orth_project(A, V):
    """
    Project A onto the tangent space of the Stiefel manifold at V.

    proj_V(A) = A - V * sym(V'A)
    """
    pro_A = A - V.dot((1/2)*(V.T.dot(A) + A.T.dot(V)))
    return pro_A


def retraction(V, theta, zeta):
    """QR-based retraction: R_V(theta*zeta) = qf(V + theta*zeta)."""
    AA = V + theta*zeta
    Q, R = np.linalg.qr(AA, mode='reduced')
    return Q


def diff_qf(Y, U):
    """
    Directional derivative of the QR factor map qf at Y in direction U.

    D qf(Y)[U] = qf(Y)*skew(qf(Y)'*U*(qf(Y)'*Y)^{-1})
                 + (I - qf(Y)*qf(Y)')*U*(qf(Y)'*Y)^{-1}
    """
    d = np.shape(Y)[0]
    Q, R = np.linalg.qr(Y, mode='reduced')
    C = np.linalg.inv(Q.T.dot(Y))
    F = Q.T.dot(U).dot(C)
    skew = np.triu(F, 1) - np.triu(F, 1).T
    diff = Q.dot(skew) + (np.eye(d) - Q.dot(Q.T)).dot(U).dot(C)
    return diff


def diff_retraction(V, X, U):
    """Directional derivative of retraction: D R_V(X)[U] = D qf(X+V)[U]."""
    Y = V + X
    diff = diff_qf(Y, U)
    return diff


# ============================================================================
# Smooth functions and their gradients (loop-based, for reference)
# ============================================================================

def L1_norm_smooth_1(V, mu):
    d, r = V.shape
    smooth_V = np.zeros((d, r))
    for i in range(d):
        for j in range(r):
            if abs(V[i, j]) <= mu:
                smooth_V[i, j] = (V[i, j]**2) / (2*mu)
            else:
                smooth_V[i, j] = abs(V[i, j]) - mu/2
    return smooth_V


def L1_norm_smooth_5(V, mu):
    d, r = V.shape
    smooth_V = np.zeros((d, r))
    for i in range(d):
        for j in range(r):
            if V[i, j] >= mu/2:
                smooth_V[i, j] = V[i, j]
            elif V[i, j] <= -mu/2:
                smooth_V[i, j] = -V[i, j]
            else:
                smooth_V[i, j] = (V[i, j]**2)/mu + mu/4
    return smooth_V


def L1_norm_smooth_1_diff(V, mu):
    d, r = V.shape
    diff_V = np.zeros((d, r))
    for i in range(d):
        for j in range(r):
            if V[i, j] >= mu:
                diff_V[i, j] = 1
            elif V[i, j] <= -mu:
                diff_V[i, j] = -1
            else:
                diff_V[i, j] = V[i, j]/mu
    return diff_V


def L1_norm_smooth_5_diff(V, mu):
    d, r = V.shape
    diff_V = np.zeros((d, r))
    for i in range(d):
        for j in range(r):
            if V[i, j] >= mu/2:
                diff_V[i, j] = 1
            elif V[i, j] <= -mu/2:
                diff_V[i, j] = -1
            else:
                diff_V[i, j] = 2*V[i, j]/mu
    return diff_V


# ============================================================================
# Smooth function dispatcher
# ============================================================================

def smooth_parameter(kind):
    """Default mu parameter for each smooth function variant."""
    return {
        '1': 0.001, '2': 0.005, '3': 0.001, '4': 0.005,
        '5': 0.001, '6': 0.001, '7': 0.001, '8': 0.001,
    }[kind]


def L1_norm_smooth(V, kind, mu):
    """Dispatch to the appropriate smooth function variant."""
    if mu == '0':
        mu = smooth_parameter(kind)
    funcs = {'1': L1_norm_smooth_1, '5': L1_norm_smooth_5}
    return funcs[kind](V, mu)


def L1_norm_smooth_diff(V, kind, mu):
    """Dispatch to the appropriate smooth function derivative."""
    if mu == '0':
        mu = smooth_parameter(kind)
    funcs = {'1': L1_norm_smooth_1_diff, '5': L1_norm_smooth_5_diff}
    return funcs[kind](V, mu)


# ============================================================================
# Objective function and gradient
# ============================================================================

def f_V(X, V, lambdm, Omega, Uplison, rho, kind, mu):
    """
    Augmented Lagrangian for the worker sub-problem.

    f(V) = -(1/2)||XV||_F^2 + lambda*sum(smooth(V))
           + tr(Omega'(V - Upsilon)) + (rho/2)||V - Upsilon||_F^2
    """
    ff = (-(1/2)*(np.linalg.norm(X.dot(V), 'fro')**2) + lambdm*sum(L1_norm_smooth(V, kind, mu))
          + np.matrix.trace(Omega.T.dot(V - Uplison)) + (rho/2)*(np.linalg.norm(V - Uplison, 'fro')**2))
    return ff


def gradient_f(X, V, lambdm, Omega, Uplison, rho, kind, mu):
    """Euclidean gradient of the augmented Lagrangian."""
    ff = (-X.T.dot(X).dot(V) + lambdm*L1_norm_smooth_diff(V, kind, mu)
          + Omega + rho*(V - Uplison))
    return ff


# ============================================================================
# Line search helpers
# ============================================================================

def phi_theta(X, V, lambdm, Omega, Uplison, theta, rho, zeta, kind, mu):
    """Objective along the retraction curve: phi(theta) = f(R_V(theta*zeta))."""
    V_next = retraction(V, theta, zeta)
    phi = f_V(X, V_next, lambdm, Omega, Uplison, rho, kind, mu)
    return phi


def diff_phi_theta(X, V, lambdm, Omega, Uplison, theta, rho, zeta, kind, mu):
    """Derivative of phi along the retraction curve."""
    V_next = retraction(V, theta, zeta)
    grad = gradient_f(X, V_next, lambdm, Omega, Uplison, rho, kind, mu)
    direction = diff_retraction(V, theta*zeta, zeta)
    diff = np.matrix.trace(grad.T * direction)
    return diff


# ============================================================================
# Worker local solver (Riemannian gradient descent with Armijo line search)
# ============================================================================

def local_solver(X, V_k, lambdm, Omega_k, Uplison_k, rho, kind, mu, max_k_worker=100):
    """
    Solve the worker sub-problem via Riemannian gradient descent
    with Armijo backtracking line search on the Stiefel manifold.
    """
    iter_k_worker = 0
    theta_k = 1
    c1 = 0.001

    while (iter_k_worker < max_k_worker) and (theta_k > 10**(-7)):
        # Euclidean gradient
        grad_f = gradient_f(X, V_k, lambdm, Omega_k, Uplison_k, rho, kind, mu)
        # Project to tangent space -> descent direction
        zeta_k = -orth_project(grad_f, V_k)

        # Armijo backtracking line search
        theta_k = 1
        phi_0 = phi_theta(X, V_k, lambdm, Omega_k, Uplison_k, 0, rho, zeta_k, kind, mu)
        phi_prime_0 = diff_phi_theta(X, V_k, lambdm, Omega_k, Uplison_k, 0, rho, zeta_k, kind, mu)
        phi_k = phi_theta(X, V_k, lambdm, Omega_k, Uplison_k, theta_k, rho, zeta_k, kind, mu)
        while phi_k > (phi_0 + c1*theta_k*phi_prime_0):
            theta_k = theta_k / 2
            phi_k = phi_theta(X, V_k, lambdm, Omega_k, Uplison_k, theta_k, rho, zeta_k, kind, mu)

        # Retraction update
        V_k = retraction(V_k, theta_k, zeta_k)
        iter_k_worker += 1

    return V_k


# ============================================================================
# Data distribution utilities
# ============================================================================

def num_dist_data(K, N, max_proportion=1/3):
    """
    Decide the number of samples per partition for non-IID splitting.

    Parameters
    ----------
    K : int - number of workers
    N : int - total number of samples
    max_proportion : float - max proportion of data per worker
    """
    np.random.seed(1)
    ratio_ls = np.zeros(K)
    for i in range(K - 1):
        share = np.random.randint(1, int(np.floor(N*max_proportion)) + 1)
        N = N - share
        ratio_ls[i] = share
    ratio_ls[-1] = N
    return ratio_ls


def distr_data(worker_nodes, data):
    """Allocate data rows to workers according to worker_nodes sizes."""
    K = np.size(worker_nodes)
    D = {}
    b = 0
    for i in range(K):
        D[i] = data[int(b):int(b + worker_nodes[i])]
        b += worker_nodes[i]
    return D


# ============================================================================
# Main ADMM loop (example usage — requires a CSV data file)
# ============================================================================

if __name__ == '__main__':
    # Parameters
    K = 5           # number of workers
    r = 1           # number of principal components
    lambdm = 250    # sparsity parameter
    rho = 1000      # ADMM penalty parameter
    kind = '5'      # smooth function variant
    mu = '0'        # use default mu for the chosen variant
    max_k_worker = 50
    max_k_master = 100

    # Load and preprocess data (replace with your dataset)
    data_file = "XXX.csv"
    if not os.path.exists(data_file):
        print(f"Data file '{data_file}' not found. Please provide your dataset.")
    else:
        XXX = pd.read_csv(data_file).values
        n, d = XXX.shape

        # Split data into K workers
        worker_size = num_dist_data(K, n)
        D = distr_data(worker_size, XXX)

        # Initialize variables
        V_k = {i: np.eye(d, r) for i in range(K)}
        Omega_k = {i: np.zeros((d, r)) for i in range(K)}
        Uplison_k = np.zeros((d, r))

        # ADMM iterations
        for iter_k_master in range(max_k_master):
            # Worker update
            for j in range(K):
                V_k[j] = local_solver(D[j], V_k[j], lambdm, Omega_k[j],
                                      Uplison_k, rho, kind, mu, max_k_worker)

            # Master update (aggregation)
            Uplison_k = (1/K) * (sum(V_k.values()) + (1/rho)*sum(Omega_k.values()))

            # Dual update
            for j in range(K):
                Omega_k[j] = Omega_k[j] + rho*(V_k[j] - Uplison_k)

        print(f"Finished after {max_k_master} iterations")
        print("Master variable (Upsilon):")
        print(Uplison_k)

        # Compare with SVD
        u, s, V_svd = svds(XXX, k=r)
        print(f"||X - X*Upsilon*Upsilon'||_2 = {np.linalg.norm(XXX - XXX.dot(Uplison_k).dot(Uplison_k.T), 2)}")
        print(f"||X - X*V_svd'*V_svd||_2     = {np.linalg.norm(XXX - XXX.dot(V_svd.T).dot(V_svd), 2)}")

