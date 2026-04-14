"""
Federated Sparse PCA (FSPCA) — Main Script

Implements the ADMM-based federated sparse PCA algorithm.
Each ADMM iteration consists of:
  1. Worker update  — Riemannian gradient descent with Wolfe line search
  2. Master update  — Aggregation + soft-thresholding (L1 proximal operator)
  3. Dual update    — Dual variable update with primal residual

Reference:
    Ciou et al., "Federated Learning for Sparse Principal Component Analysis",
    IEEE BigData 2023. arXiv:2311.08677
"""
import os
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Modified scipy line search supporting matrix-valued gradients
from linesearch_muti_dem import line_search_wolfe2 as line_search
from main_function import (
    objective_function, objective_function_gradient,
    orth_project, grad_f_retraction, retraction, cosine_similarity
)

path = os.getcwd()


# ============================================================================
# Data preparation
# ============================================================================

def random_split_data(data, num, random_state=42):
    """
    Shuffle, split, normalize data, and compute covariance matrices.

    Parameters
    ----------
    data : DataFrame - input dataset
    num : int - number of partitions (workers)
    random_state : int - random seed for reproducibility

    Returns
    -------
    normalize_data : list of ndarray - normalized data partitions (varying sizes)
    two_norm_data : ndarray (num, d, d) - covariance matrices X_k = data_k' @ data_k
    """
    shuffled_df = data.sample(frac=1, random_state=random_state).reset_index(drop=True)
    split_data = np.array_split(shuffled_df, num)
    # Normalize each partition: zero mean, unit std
    normalize_data = [((item - item.mean()) / item.std()).values for item in split_data]
    # Compute covariance matrix for each partition
    two_norm_data = [item.T @ item for item in normalize_data]
    return normalize_data, np.array(two_norm_data)


# Load dataset
data = pd.read_csv(os.path.join(path, 'data', 'breast_cancer.csv'))
data = data[data.columns[1:-1]]

# Append random noise features to test sparsity recovery
rand_data_num = 500
rand_data_col = ['rand_col_' + str(i) for i in range(1, rand_data_num + 1)]
random_features = pd.DataFrame(np.random.rand(data.shape[0], rand_data_num), columns=rand_data_col)
data = pd.concat([data, random_features], axis=1)
print('Data dimension after adding random features:', data.shape)

# Split data into K partitions and normalize
K = 5
split_data, two_norm_data = random_split_data(data, K, random_state=42)
# Verify normalization
for item in split_data:
    print(np.mean(item), np.std(item), np.shape(item))


# ============================================================================
# Variable initialization
# ============================================================================

# Initialize worker variables with local PCA
pca_num = 2
worker_varaible_list = []
for item in split_data:
    pca = PCA(n_components=pca_num)
    pca.fit(item)
    worker_varaible_list.append(pca.components_.T)
worker_varaible_list = np.array(worker_varaible_list)
print('Worker variable shape:', np.shape(worker_varaible_list[0]))

d = np.shape(worker_varaible_list[0])[0]
r = np.shape(worker_varaible_list[0])[1]

# Initialize dual variables with identity
dual_varaible_list = np.array([np.eye(d, r) for _ in range(K)])

# Initialize master variable with zeros
master_varaible = np.zeros((d, r))


# ============================================================================
# ADMM parameters
# ============================================================================

lambdm_worker = 20        # L1 regularization weight for worker sub-problem
lambdm_master = 20        # L1 regularization weight for master soft-thresholding
rho = 5000                # ADMM penalty parameter
worker_tol = 1 - 10**(-3) # Cosine similarity threshold for worker convergence
tol_rel = 10**(-2)        # Relative tolerance for ADMM convergence
max_iter = 300            # Maximum ADMM outer iterations
max_worker_iter = 20      # Maximum Riemannian gradient descent steps per worker

# History storage
worker_history = []
dual_history = []
master_history = []


# ============================================================================
# ADMM main loop
# ============================================================================

for all_iter in range(max_iter):

    # --- Step 1: Worker update (Riemannian gradient descent) ---
    for i in range(K):
        cov = two_norm_data[i]
        worker_varaible = worker_varaible_list[i]
        dual_varaible = dual_varaible_list[i]

        start = time.time()
        for worker_iter in range(max_worker_iter):
            # Compute Euclidean gradient and project onto tangent space
            grad_f = objective_function_gradient(
                worker_varaible, cov, dual_varaible, master_varaible, lambdm_worker, rho)
            zeta = -orth_project(grad_f, worker_varaible)

            # Wolfe line search for step size
            alpha = line_search(
                objective_function, grad_f_retraction, worker_varaible, zeta,
                args=(worker_varaible, cov, dual_varaible, master_varaible, lambdm_worker, rho))
            if alpha[0]:
                thrta = alpha[0]
            else:
                thrta = 0.001
                print('Line search did not converge, using default step size.')

            # Update worker variable with retraction back onto the Stiefel manifold
            worker_varaible_old = worker_varaible
            worker_varaible = retraction(worker_varaible, thrta, zeta)

            # Check worker convergence via cosine similarity
            worker_res = cosine_similarity(worker_varaible, worker_varaible_old)
            if worker_res < worker_tol:
                break

        worker_varaible_list[i] = worker_varaible
        end = time.time()
        print(format(end - start))

    # --- Step 2: Master update (aggregation + soft-thresholding) ---
    master_varaible_old = master_varaible
    master_varaible = np.sum(worker_varaible_list + dual_varaible_list / rho, axis=0) / K
    # Soft-thresholding (proximal operator of L1 norm)
    master_varaible = np.sign(master_varaible) * np.maximum(
        np.abs(master_varaible) - lambdm_master / (K * rho), 0)

    # --- Step 3: Dual variable update ---
    dual_varaible_list += rho * (worker_varaible_list - master_varaible)

    # Record history
    worker_history.append(worker_varaible_list.copy())
    dual_history.append(dual_varaible_list.copy())
    master_history.append(master_varaible.copy())

    # Print convergence diagnostics
    print(f'Iteration {all_iter}')
    for i in range(len(worker_varaible_list)):
        for j in range(i + 1, len(worker_varaible_list)):
            print(f"  Cosine sim worker {i+1} vs {j+1}: "
                  f"{cosine_similarity(worker_varaible_list[i], worker_varaible_list[j])}")
    for i in range(len(worker_varaible_list)):
        print(f"  Cosine sim worker {i+1} vs master: "
              f"{cosine_similarity(worker_varaible_list[i], master_varaible)}")

    # --- ADMM convergence check ---
    # Primal residual
    prime_stop = np.sqrt(np.sum(
        [np.linalg.norm(item, 'fro') for item in worker_varaible_list - master_varaible]))
    # Dual residual
    dual_stop = (rho * np.linalg.norm(master_varaible - master_varaible_old, 'fro'))**2

    # Primal tolerance
    worker_sum_norm = np.sum([np.linalg.norm(item, 'fro')**2 for item in worker_varaible_list])
    tol_pri = tol_rel * max(worker_sum_norm, K * np.linalg.norm(master_varaible, 'fro')**2)
    # Dual tolerance
    tol_dual = tol_rel * np.sum([np.linalg.norm(item, 'fro')**2 for item in dual_varaible_list])

    if prime_stop < tol_pri and dual_stop < tol_dual:
        print(f'Converged at iteration {all_iter}')
        break



