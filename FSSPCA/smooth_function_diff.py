# -*- coding: utf-8 -*-
"""
Vectorized derivatives of smooth approximations of |x| (L1 norm).

Corresponds to the functions in smooth_function.py.
"""
import numpy as np
from scipy.special import erf


def L1_norm_smooth_diff_1(vector, mu=0.001):
    """Derivative of variant 1: Huber-like piecewise linear."""
    return np.where(np.abs(vector) <= mu, vector/mu,
                    np.sign(vector))


def L1_norm_smooth_diff_2(vector, mu=0.005):
    """Derivative of variant 2: tanh."""
    return np.sinh(vector/mu)/np.cosh(vector/mu)


def L1_norm_smooth_diff_3(vector, mu=0.001):
    """Derivative of variant 3."""
    mu_squ = mu**2
    return vector/np.sqrt(np.square(vector) + mu_squ)


def L1_norm_smooth_diff_4(vector, mu=0.005):
    """Derivative of variant 4. Note: mu too small may cause overflow."""
    p_exp_vector = np.exp(vector/mu)
    n_exp_vector = np.exp(-vector/mu)
    return (p_exp_vector/(1 + p_exp_vector)) - (n_exp_vector/(1 + n_exp_vector))


def L1_norm_smooth_diff_5(vector, mu=0.01):
    """Derivative of variant 5 (recommended): piecewise linear."""
    return np.where(np.abs(vector) < mu/2, 2*vector/mu,
                    np.sign(vector))


def L1_norm_smooth_diff_6(vector, mu=0.001):
    """Derivative of variant 6."""
    four_mu_squ = 4*mu**2
    return vector/np.sqrt(four_mu_squ + np.square(vector))


def L1_norm_smooth_diff_7(vector, mu=0.001):
    """Derivative of variant 7: cubic polynomial in [-mu, mu], sign outside."""
    two_mu_3 = 2*mu**3
    two_mu = 2*mu
    return np.where(np.abs(vector) <= mu, -np.power(vector, 3)/two_mu_3 + 3*vector/two_mu,
                    np.sign(vector))


def L1_norm_smooth_diff_8(vector, mu=0.001):
    """Derivative of variant 8: erf-based."""
    s2_mu = np.sqrt(2)*mu
    inverse_mu = 1/mu
    s_2_pi = np.sqrt(2/np.pi)
    two_squ_mu = 2*np.square(mu)
    return erf(vector/s2_mu) + (inverse_mu - mu)*(s_2_pi*vector*np.exp(-vector/two_squ_mu))











