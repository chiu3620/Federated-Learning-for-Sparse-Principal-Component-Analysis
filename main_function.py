# -*- coding: utf-8 -*-
"""
Core functions for Federated Sparse PCA (FSPCA).

Includes objective function, gradient, Riemannian optimization utilities
(orthogonal projection, QR-based retraction on the Stiefel manifold),
and line search helper functions (phi, phi').

Reference:
    Ciou et al., "Federated Learning for Sparse Principal Component Analysis",
    IEEE BigData 2023. arXiv:2311.08677
"""
import numpy as np
from smooth_function import absolute_smooth_5, absolute_smooth_diff_5
import scipy


def cosine_similarity(A, B):
    """Compute cosine similarity between two matrices (flattened)."""
    A_vec = np.ravel(A)
    B_vec = np.ravel(B)
    cos_sim = np.dot(A_vec, B_vec) / (np.linalg.norm(A_vec) * np.linalg.norm(B_vec))
    return cos_sim


def objective_function(worker_varaible, cov, dual_varaible, master_varaible, lambdm_worker, rho):
    """
    Augmented Lagrangian for the worker sub-problem.

    f(V) = -tr(V'XV)/2 + tr(Ω'(V-Υ)) + (ρ/2)||V-Υ||_F^2 + λ·Σsmooth(|V|)

    Parameters
    ----------
    worker_varaible : ndarray (d, r) - worker variable V
    cov : ndarray (d, d) - covariance matrix X = data' @ data
    dual_varaible : ndarray (d, r) - dual variable Ω
    master_varaible : ndarray (d, r) - master variable Υ
    lambdm_worker : float - L1 regularization weight λ
    rho : float - ADMM penalty parameter ρ
    """
    out = -np.matrix.trace(worker_varaible.T @ cov @ worker_varaible)/2 + \
          np.matrix.trace(dual_varaible.T @ (worker_varaible - master_varaible)) + \
          rho*(np.linalg.norm(worker_varaible - master_varaible, 'fro')**2)/2 + \
          lambdm_worker*np.sum(absolute_smooth_5(worker_varaible))
    return out


def objective_function_gradient(worker_varaible, cov, dual_varaible, master_varaible, lambdm_worker, rho):
    """
    Euclidean gradient of the augmented Lagrangian.

    ∇f(V) = -XV + Ω + ρ(V-Υ) + λ·smooth'(V)
    """
    out = -cov @ worker_varaible + dual_varaible + rho*(worker_varaible - master_varaible) + \
          lambdm_worker*absolute_smooth_diff_5(worker_varaible)
    return out


def orth_project(riemann_gradient, orth, G=None):
    """
    Project a matrix onto the tangent space of the (generalized) Stiefel manifold.

    proj_V(Z) = Z - V · sym(V' G Z)

    Parameters
    ----------
    riemann_gradient : ndarray (d, r) - the matrix to project (gradient)
    orth : ndarray (d, r) - the current point on the Stiefel manifold V
    G : ndarray (d, d), optional - metric matrix for generalized Stiefel manifold
    """
    if G:
        temp = orth.T @ G @ riemann_gradient
    else:
        temp = orth.T @ riemann_gradient
    pro_gradient = riemann_gradient - orth @ ((temp + temp.T)/2)
    return pro_gradient


def retraction(V, theta, zeta, G=None):
    """
    QR-based retraction for the (generalized) Stiefel manifold.

    R_V(θ·ζ) = qf(V + θ·ζ), where qf is the Q-factor of QR decomposition.

    Parameters
    ----------
    V : ndarray (d, r) - current point on the manifold
    theta : float - step size
    zeta : ndarray (d, r) - descent direction
    G : ndarray (d, d), optional - metric matrix for generalized Stiefel manifold
    """
    Y = V + theta*zeta
    if G:
        R = scipy.linalg.cholesky(Y.T @ G @ Y).T
    else:
        R = scipy.linalg.cholesky(Y.T @ Y).T
    Q = Y @ np.linalg.inv(R)
    return Q


def grad_f_retraction(worker_varaible, cov, dual_varaible, master_varaible, lambdm_worker, rho):
    """
    Compute the Riemannian gradient after retracting the worker variable
    back onto the Stiefel manifold via QR decomposition.

    Used as the gradient function (fprime) for Wolfe line search.
    """
    Q, _ = np.linalg.qr(worker_varaible, mode='reduced')
    grad = objective_function_gradient(Q, cov, dual_varaible, master_varaible, lambdm_worker, rho)
    out = orth_project(grad, Q)
    return out


def phi_theta(theta, zeta, worker_varaible, cov, dual_varaible, master_varaible, lambdm_worker, rho):
    """
    Objective function along the retraction curve: φ(θ) = f(R_V(θ·ζ)).

    Used for line search.
    """
    if theta == 0:
        worker_varaible_next = worker_varaible
    else:
        worker_varaible_next = retraction(worker_varaible, theta, zeta)
    out = objective_function(worker_varaible_next, cov, dual_varaible, master_varaible, lambdm_worker, rho)
    return out


def phi_function_diff(theta, zeta, worker_varaible, cov, dual_varaible, master_varaible, lambdm_worker, rho):
    """
    Derivative of φ along the retraction curve: φ'(θ) = tr(proj∇f(R_V(θ·ζ))' · ζ).

    Used for line search.
    """
    if theta == 0:
        worker_varaible_next = worker_varaible
    else:
        worker_varaible_next = retraction(worker_varaible, theta, zeta)
    # Compute gradient at the retracted point
    grad_V = objective_function_gradient(worker_varaible_next, cov, dual_varaible, master_varaible, lambdm_worker, rho)
    # Project gradient onto the tangent space at the retracted point
    grad_V = orth_project(grad_V, worker_varaible_next)
    phi_diff = np.matrix.trace(grad_V.T @ grad_V)
    return phi_diff


