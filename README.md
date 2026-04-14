# Federated Learning for Sparse Principal Component Analysis (FSPCA)

Python implementation of the paper:

> **Federated Learning for Sparse Principal Component Analysis**
> Sin Cheng Ciou, Pin Jui Chen, Elvin Y. Tseng, Yuh-Jye Lee
> *IEEE BigData 2023, Sorrento, Italy*
> [arXiv:2311.08677](https://arxiv.org/abs/2311.08677)

## Overview

This project implements a **Federated Sparse PCA (FSPCA)** algorithm that combines:

1. **Federated Learning** — Data stays on local clients (workers). Only model updates are exchanged with a central server (master), preserving data privacy.
2. **Sparse PCA** — Attains sparse principal component loadings while maximizing explained variance, improving interpretability.
3. **ADMM (Alternating Direction Method of Multipliers)** — The consensus optimization framework used to coordinate workers and the master.
4. **Riemannian Optimization on the Stiefel Manifold** — Worker sub-problems are solved via gradient descent on the orthogonal constraint manifold, using QR-based retraction and Wolfe line search.
5. **Smooth Approximation of L1 Norm** — A smooth surrogate of the absolute value function is used to enable gradient-based optimization for the sparsity-inducing penalty.

## Algorithm

The ADMM iteration consists of three steps per round:

| Step | Description |
|------|-------------|
| **Worker Update** | Each worker minimizes its local augmented Lagrangian via Riemannian gradient descent with Wolfe line search on the Stiefel manifold. |
| **Master Update** | The master aggregates worker variables and applies soft-thresholding (proximal operator of L1 norm) to promote sparsity. |
| **Dual Update** | Dual variables are updated with the primal residual. |

Convergence is checked using primal and dual residual tolerances.

## Project Structure

```
FSPCA/
├── main.py                  # Main script — runs the full FSPCA algorithm
├── main_function.py         # Core functions: objective, gradient, retraction,
│                            #   orthogonal projection, cosine similarity
├── smooth_function.py       # Smooth approximations of |x| and their derivatives
│                            #   (8 variants; variant 5 is used by default)
├── linesearch_muti_dem.py   # Modified scipy Wolfe line search supporting
│                            #   matrix-valued gradients (trace inner product)
├── data/
│   └── breast_cancer.csv    # Wisconsin Diagnostic Breast Cancer (WDBC) dataset
├── FSSPCA/                  # Smooth function analysis and early prototype
│   ├── Smooth Federated PCA.py   # Early standalone prototype (from Jupyter notebook)
│   ├── smooth_function.py        # Vectorized smooth functions with docstrings
│   ├── smooth_function_diff.py   # Vectorized smooth function derivatives
│   ├── smooth_function_test.py   # Visualization: plots smooth functions with various μ
│   └── smooth.png                # Output plot of smooth approximations
└── README.md
```

## Requirements

- Python >= 3.8
- NumPy
- Pandas
- scikit-learn
- SciPy

Install dependencies:

```bash
pip install numpy pandas scikit-learn scipy
```

## Usage

Run the main FSPCA algorithm:

```bash
python main.py
```

### Key Parameters (in `main.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `K` | 5 | Number of federated workers (data partitions) |
| `pca_num` | 2 | Number of principal components to extract |
| `rand_data_num` | 500 | Number of random noise features appended to test sparsity |
| `lambdm_worker` | 20 | L1 regularization weight for worker sub-problem |
| `lambdm_master` | 20 | L1 regularization weight for master soft-thresholding |
| `rho` | 5000 | ADMM penalty parameter |
| `max_iter` | 300 | Maximum number of ADMM outer iterations |
| `max_worker_iter` | 20 | Maximum Riemannian gradient descent steps per worker |
| `tol_rel` | 1e-2 | Relative tolerance for ADMM convergence |

## Module Details

### `main_function.py`

- `objective_function` — Augmented Lagrangian: `-tr(V'XV)/2 + tr(Ω'(V-Υ)) + (ρ/2)‖V-Υ‖² + λΣsmooth(|V|)`
- `objective_function_gradient` — Euclidean gradient of the augmented Lagrangian
- `orth_project` — Projection onto the tangent space of the (generalized) Stiefel manifold
- `retraction` — QR-based retraction to pull iterates back onto the Stiefel manifold
- `cosine_similarity` — Matrix cosine similarity (flattened) for convergence monitoring

### `smooth_function.py`

Provides 8 smooth approximations of `|x|` and their derivatives. The default variant used is **variant 5** (piecewise quadratic):

- When `|x| ≤ μ/2`: `smooth(x) = x²/μ + μ/4`
- Otherwise: `smooth(x) = |x|`

### `linesearch_muti_dem.py`

A modified version of `scipy.optimize.line_search` that supports **matrix-valued** variables and gradients. The inner product is replaced by `tr(G'P)` instead of the vector dot product.

## Citation

```bibtex
@inproceedings{ciou2023federated,
  title={Federated Learning for Sparse Principal Component Analysis},
  author={Ciou, Sin Cheng and Chen, Pin Jui and Tseng, Elvin Y. and Lee, Yuh-Jye},
  booktitle={2023 IEEE International Conference on Big Data (BigData)},
  year={2023},
  organization={IEEE}
}
```

## License

This project is for academic and research purposes.

