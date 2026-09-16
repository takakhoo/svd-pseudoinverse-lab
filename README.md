# Singular Value Decomposition and the Moore–Penrose Pseudoinverse

A focused linear-algebra notebook that decomposes matrices with NumPy and
constructs the Moore–Penrose pseudoinverse from the singular factors. The goal
is to make the mechanics of SVD inspectable before using it in larger machine
learning or signal-processing systems.

## What it covers

- The factorization \(A=U\Sigma V^T\)
- Singular values and orthogonal factors
- Pseudoinverse construction by inverting nonzero singular values
- Numerical checks on small matrices

## Quick start

```bash
git clone https://github.com/takakhoo/svd-pseudoinverse-lab.git
cd svd-pseudoinverse-lab
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
jupyter lab "Singular Value Decomposition.ipynb"
```

[Open the executed notebook](Singular%20Value%20Decomposition.ipynb)

## Verification

The notebook was executed end to end on September 16, 2026. It reconstructs an
input matrix from its factors and compares the custom pseudoinverse calculation
with NumPy's reference implementation. The notebook is deliberately small so
each operation can be checked directly.

## Scope

This is an educational calculation, not a replacement for the numerically
robust routines in `numpy.linalg` or `scipy.linalg`.
