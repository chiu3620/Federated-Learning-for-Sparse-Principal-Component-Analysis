"""
Vectorized smooth approximations of |x| (L1 norm).

Eight variants are provided. Variant 5 (piecewise quadratic) is recommended.
"""
import numpy as np
from scipy.special import erf


def L1_norm_smooth_1(vector, mu=0.001):
    """Variant 1: Huber-like piecewise quadratic/linear."""
    double_mu = 2*mu
    helf_mu = mu/2
    return np.where(np.abs(vector) <= mu, np.square(vector)/double_mu,
                    np.abs(vector) - helf_mu)


def L1_norm_smooth_2(vector, mu=0.005):
    """Variant 2: log-cosh approximation."""
    return mu*np.log(np.cosh(vector/mu))


def L1_norm_smooth_3(vector, mu=0.001):
    """Variant 3: sqrt(x^2 + mu^2) - mu."""
    mu_squ = mu**2
    return np.sqrt(np.square(vector) + mu_squ) - mu


def L1_norm_smooth_4(vector, mu=0.005):
    """Variant 4: softplus-based. Note: mu too small may cause overflow."""
    return mu*(np.log(1 + np.exp(-vector/mu)) + np.log(1 + np.exp(vector/mu)))


def L1_norm_smooth_5(vector, mu=0.01):
    """Variant 5 (recommended): piecewise quadratic — exact outside [-mu/2, mu/2]."""
    helf_helf_mu = mu/4
    return np.where(vector >= mu/2, vector,
                    np.where(vector <= -mu/2,
                              -vector, np.square(vector)/mu + helf_helf_mu))


def L1_norm_smooth_6(vector, mu=0.001):
    """Variant 6: sqrt(4*mu^2 + x^2)."""
    four_mu_squ = 4*mu**2
    return np.sqrt(four_mu_squ + np.square(vector))


def L1_norm_smooth_7(vector, mu=0.001):
    """Variant 7: quartic polynomial in [-mu, mu], exact outside."""
    eight_mu_3 = 8*mu**3
    four_mu = 4*mu
    three_mu_8 = 3*mu/8
    return np.where(vector >= mu, vector,
                    np.where(vector <= -mu,
                              -vector, -np.power(vector, 4)/eight_mu_3 + 3*np.square(vector)/four_mu + three_mu_8))


def L1_norm_smooth_8(vector, mu=0.001):
    """Variant 8: erf-based approximation."""
    s2_mu = np.sqrt(2)*mu
    s_2_pi = np.sqrt(2/np.pi)
    two_squ_mu = 2*np.square(mu)
    return vector*erf(vector/s2_mu) + s_2_pi*mu*np.exp(-np.square(vector)/two_squ_mu)