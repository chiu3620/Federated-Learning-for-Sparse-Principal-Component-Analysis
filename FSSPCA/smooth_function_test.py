# -*- coding: utf-8 -*-
"""
Visualization script for smooth approximations of |x|.

Plots each of the 8 smooth function variants with various mu values
alongside the true |x| for comparison.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf


# ============================================================================
# Smooth function definitions (self-contained for standalone use)
# ============================================================================

def L1_norm_smooth_1(vector, mu=0.001):
    """Variant 1: Huber-like piecewise quadratic/linear."""
    return np.where(np.abs(vector) <= mu, np.square(vector)/(2*mu),
                    np.abs(vector) - mu/2)


def L1_norm_smooth_2(vector, mu=0.001):
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


# ============================================================================
# Plotting
# ============================================================================

mu_values = [10, 1, 0.1, 0.01, 0.001, 0.0001]

smooth_functions = {
    'Variant 1 (Huber-like)': L1_norm_smooth_1,
    'Variant 2 (log-cosh)': L1_norm_smooth_2,
    'Variant 3 (sqrt)': L1_norm_smooth_3,
    'Variant 4 (softplus)': L1_norm_smooth_4,
    'Variant 5 (piecewise quadratic)': L1_norm_smooth_5,
    'Variant 6 (sqrt shifted)': L1_norm_smooth_6,
    'Variant 7 (quartic)': L1_norm_smooth_7,
    'Variant 8 (erf-based)': L1_norm_smooth_8,
}

x = np.linspace(-1, 1, 100000)

fig, axes = plt.subplots(2, 4, figsize=(20, 8))
axes = axes.flatten()

for idx, (name, func) in enumerate(smooth_functions.items()):
    ax = axes[idx]
    for mu in mu_values:
        try:
            ax.plot(x, func(x, mu=mu), label=f'μ={mu}')
        except (OverflowError, FloatingPointError):
            pass  # Skip mu values that cause numerical issues (e.g., variant 4)
    ax.plot(x, np.abs(x), 'k--', label='|x|', linewidth=2)
    ax.set_title(name)
    ax.legend(fontsize=7)
    ax.set_xlabel('x')
    ax.set_ylabel('smooth(x)')

plt.suptitle('Smooth Approximations of |x| with Various μ Values', fontsize=14)
plt.tight_layout()
plt.savefig('smooth.png', dpi=150)
plt.show()
