"""Thresholded SVD pseudoinverse and a reproducible inverse-problem experiment."""
import argparse
import json
import platform
from pathlib import Path
import numpy as np


def pseudoinverse(matrix, rcond=1e-15):
    """Return the truncated Moore-Penrose inverse, including rectangular/complex A.

    Singular values <= rcond * largest singular value are discarded. This is a
    regularized inverse when the threshold deliberately removes nonzero values.
    """
    a = np.asarray(matrix)
    if a.ndim != 2 or 0 in a.shape or not np.isfinite(a).all():
        raise ValueError("a nonempty, finite 2-D matrix is required")
    if not np.isfinite(rcond) or rcond < 0:
        raise ValueError("rcond must be finite and nonnegative")
    u, s, vh = np.linalg.svd(a, full_matrices=False)
    reciprocal = np.zeros_like(s)
    np.divide(1.0, s, out=reciprocal, where=s > rcond * s[0])
    return (vh.conj().T * reciprocal) @ u.conj().T


def penrose_residuals(a, inverse):
    """Scale-normalized residuals for the four Moore-Penrose identities."""
    def relative(actual, expected):
        return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1.0))
    left, right = a @ inverse, inverse @ a
    return {"A_Ap_A": relative(left @ a, a), "Ap_A_Ap": relative(right @ inverse, inverse),
            "A_Ap_hermitian": relative(left, left.conj().T),
            "Ap_A_hermitian": relative(right, right.conj().T)}


def experiment(seed=7):
    rng = np.random.default_rng(seed)
    u, _ = np.linalg.qr(rng.normal(size=(60, 20)))
    v, _ = np.linalg.qr(rng.normal(size=(20, 20)))
    singular_values = np.geomspace(1.0, 1e-10, 20)
    a = (u * singular_values) @ v.T
    truth = rng.normal(size=20)
    clean = a @ truth
    observed = clean + rng.normal(scale=1e-4, size=60)
    rows = []
    for cutoff in np.logspace(-15, -1, 29):
        estimate = pseudoinverse(a, cutoff) @ observed
        rows.append({"rcond": float(cutoff), "rank": int(np.sum(singular_values > cutoff)),
                     "relative_solution_error": float(np.linalg.norm(estimate - truth) / np.linalg.norm(truth)),
                     "relative_observation_residual": float(np.linalg.norm(a @ estimate - observed) / np.linalg.norm(observed))})
    cases = {"square": rng.normal(size=(8, 8)), "tall": rng.normal(size=(12, 5)),
             "wide": rng.normal(size=(5, 12)), "rank_deficient": np.outer([1., 2., 3.], [4., 5.]),
             "zero": np.zeros((4, 6)), "complex": rng.normal(size=(4, 7)) + 1j*rng.normal(size=(4, 7))}
    validation = {name: penrose_residuals(matrix, pseudoinverse(matrix)) for name, matrix in cases.items()}
    return {"seed": seed, "python": platform.python_version(), "numpy": np.__version__,
            "shape": [60, 20], "condition_number": 1e10, "noise_standard_deviation": 1e-4,
            "sweep": rows, "penrose_checks": validation}, singular_values


def export(output, seed=7):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    report, spectrum = experiment(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    axes[0].semilogy(np.arange(1, 21), spectrum, "o-", color="#385a91", markersize=3)
    axes[0].set(title="Ill-conditioned forward operator", xlabel="Singular-value index", ylabel="Singular value")
    cutoffs = [r["rcond"] for r in report["sweep"]]
    axes[1].loglog(cutoffs, [r["relative_solution_error"] for r in report["sweep"]], color="#ae573d")
    axes[1].set(title="Recovery error vs. cutoff", xlabel="Relative cutoff", ylabel="Relative solution error")
    axes[2].loglog(cutoffs, [r["relative_observation_residual"] for r in report["sweep"]], color="#287b70")
    axes[2].set(title="Fitting noise is not recovery", xlabel="Relative cutoff", ylabel="Relative observation residual")
    for ax in axes:
        ax.grid(alpha=.15)
    fig.savefig(output / "noise-amplification.png", dpi=180)
    plt.close(fig)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    result = export(args.output, args.seed)
    print(json.dumps({"unregularized": result["sweep"][0], "cutoff_1e-4": result["sweep"][22],
                      "max_penrose_residual": max(v for row in result["penrose_checks"].values() for v in row.values())}, indent=2))
