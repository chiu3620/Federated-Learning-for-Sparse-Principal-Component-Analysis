"""
Smooth approximations of |x| and their derivatives.

Eight variants are provided. Variant 5 (piecewise quadratic) is used as the
default in the FSPCA algorithm.

Reference:
    Ciou et al., "Federated Learning for Sparse Principal Component Analysis",
    IEEE BigData 2023. arXiv:2311.08677
"""
import numpy as np
from scipy.special import erf


# ============================================================================
# Smooth approximations of |x|
# ============================================================================

def absolute_smooth_1(vector, mu=0.001):
    """Variant 1: Huber-like piecewise quadratic/linear."""
    double_mu = 2*mu
    helf_mu = mu/2
    return np.where(np.abs(vector) <= mu, np.square(vector)/double_mu,
                    np.abs(vector) - helf_mu)


def absolute_smooth_2(vector, mu=0.005):
    """Variant 2: log-cosh approximation."""
    return mu*np.log(np.cosh(vector/mu))


def absolute_smooth_3(vector, mu=0.001):
    """Variant 3: sqrt(x^2 + mu^2) - mu."""
    mu_squ = mu**2
    return np.sqrt(np.square(vector) + mu_squ) - mu


def absolute_smooth_4(vector, mu=0.005):
    """Variant 4: softplus-based approximation. Note: mu too small may cause overflow."""
    return mu*(np.log(1 + np.exp(-vector/mu)) + np.log(1 + np.exp(vector/mu)))


def absolute_smooth_5(vector, mu=2*10**(-6)):
    """Variant 5 (default): piecewise quadratic — exact outside [-mu/2, mu/2]."""
    helf_mu = mu/2
    quarter_mu = mu/4
    return np.where(np.abs(vector) <= helf_mu, np.square(vector)/mu + quarter_mu,
                    np.abs(vector))


def absolute_smooth_6(vector, mu=0.001):
    """Variant 6: sqrt(4*mu^2 + x^2)."""
    four_mu_squ = 4*mu**2
    return np.sqrt(four_mu_squ + np.square(vector))


def absolute_smooth_7(vector, mu=0.001):
    """Variant 7: quartic polynomial in [-mu, mu], exact outside."""
    eight_mu_3 = 8*mu**3
    four_mu = 4*mu
    three_mu_8 = 3*mu/8
    return np.where(np.abs(vector) <= mu, -np.power(vector, 4)/eight_mu_3 + 3*np.square(vector)/four_mu + three_mu_8,
                    np.abs(vector))


def absolute_smooth_8(vector, mu=0.001):
    """Variant 8: erf-based approximation."""
    s2_mu = np.sqrt(2)*mu
    s_2_pi = np.sqrt(2/np.pi)
    two_squ_mu = 2*np.square(mu)
    return vector*erf(vector/s2_mu) + s_2_pi*mu*np.exp(-np.square(vector)/two_squ_mu)


# ============================================================================
# Derivatives of smooth approximations
# ============================================================================

def absolute_smooth_diff_1(vector, mu=0.001):
    """Derivative of variant 1."""
    return np.where(np.abs(vector) <= mu, vector/mu,
                    np.sign(vector))


def absolute_smooth_diff_2(vector, mu=0.005):
    """Derivative of variant 2."""
    return np.sinh(vector/mu)/np.cosh(vector/mu)


def absolute_smooth_diff_3(vector, mu=0.001):
    """Derivative of variant 3."""
    mu_squ = mu**2
    return vector/np.sqrt(np.square(vector) + mu_squ)


def absolute_smooth_diff_4(vector, mu=0.005):
    """Derivative of variant 4."""
    p_exp_vector = np.exp(vector/mu)
    n_exp_vector = np.exp(-vector/mu)
    return (p_exp_vector/(1 + p_exp_vector)) - (n_exp_vector/(1 + n_exp_vector))


def absolute_smooth_diff_5(vector, mu=2*10**(-6)):
    """Derivative of variant 5 (default)."""
    return np.where(np.abs(vector) < mu/2, 2*vector/mu,
                    np.sign(vector))


def absolute_smooth_diff_6(vector, mu=0.001):
    """Derivative of variant 6."""
    four_mu_squ = 4*mu**2
    return vector/np.sqrt(four_mu_squ + np.square(vector))


def absolute_smooth_diff_7(vector, mu=0.001):
    """Derivative of variant 7."""
    two_mu_3 = 2*mu**3
    two_mu = 2*mu
    return np.where(np.abs(vector) <= mu, -np.power(vector, 3)/two_mu_3 + 3*vector/two_mu,
                    np.sign(vector))


def absolute_smooth_diff_8(vector, mu=0.001):
    """Derivative of variant 8."""
    s2_mu = np.sqrt(2)*mu
    inverse_mu = 1/mu
    s_2_pi = np.sqrt(2/np.pi)
    two_squ_mu = 2*np.square(mu)
    return erf(vector/s2_mu) + (inverse_mu - mu)*(s_2_pi*vector*np.exp(-vector/two_squ_mu))
