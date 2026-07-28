"""Empirical CP-rank bound for the S4 relational tensor.

The S4 kernel exposes operand interactions only through

    q_{pr} = P_{pr} * (sum_i L_{ir} a_i) * (sum_j R_{jr} b_j)

which is algebraically equivalent to contracting the operands with a tensor

    T_{pij} = sum_r P_{pr} L_{ir} R_{jr}

of CP rank at most R (the configured `shared_rank`).

Unsigned multiplication needs the partial-product sums

    s_p = sum_{i+j=p} a_i b_j   <=>   T_{pij} = 1[i + j = p],

the (truncated) polynomial-multiplication tensor. This script measures the
best achievable rank-R CP approximation error of that tensor by alternating
least squares with random restarts, for R = 1..2W. A sharp drop to numerical
zero locates the exact rank and therefore the minimum `shared_rank` at which
multiplication is representable at all.

Addition and subtraction are included as controls: they are linear in each
operand bit and need no relational tensor, which is why S4 already solves them.
"""

from __future__ import annotations

import numpy as np

RNG_SEED = 20260728


def multiplication_tensor(width: int) -> np.ndarray:
    """T[p, i, j] = 1 iff i + j == p, for output bits p in [0, width)."""
    tensor = np.zeros((width, width, width))
    for i in range(width):
        for j in range(width):
            if i + j < width:
                tensor[i + j, i, j] = 1.0
    return tensor


def als_cp(
    target: np.ndarray,
    rank: int,
    iterations: int = 4000,
    restarts: int = 6,
    seed: int = RNG_SEED,
) -> float:
    """Return relative Frobenius error of the best rank-`rank` CP fit."""
    rng = np.random.default_rng(seed)
    dim_p, dim_i, dim_j = target.shape
    norm = np.linalg.norm(target)
    best = np.inf

    # Unfoldings along each mode, computed once.
    unfold_p = target.reshape(dim_p, dim_i * dim_j)
    unfold_i = np.transpose(target, (1, 0, 2)).reshape(dim_i, dim_p * dim_j)
    unfold_j = np.transpose(target, (2, 0, 1)).reshape(dim_j, dim_p * dim_i)

    for _ in range(restarts):
        factor_p = rng.standard_normal((dim_p, rank))
        factor_i = rng.standard_normal((dim_i, rank))
        factor_j = rng.standard_normal((dim_j, rank))

        for _ in range(iterations):
            # Solve each factor in closed form holding the others fixed.
            khatri = (factor_i[:, None, :] * factor_j[None, :, :]).reshape(-1, rank)
            gram = (factor_i.T @ factor_i) * (factor_j.T @ factor_j)
            factor_p = np.linalg.solve(
                gram + 1e-10 * np.eye(rank), (unfold_p @ khatri).T
            ).T

            khatri = (factor_p[:, None, :] * factor_j[None, :, :]).reshape(-1, rank)
            gram = (factor_p.T @ factor_p) * (factor_j.T @ factor_j)
            factor_i = np.linalg.solve(
                gram + 1e-10 * np.eye(rank), (unfold_i @ khatri).T
            ).T

            khatri = (factor_p[:, None, :] * factor_i[None, :, :]).reshape(-1, rank)
            gram = (factor_p.T @ factor_p) * (factor_i.T @ factor_i)
            factor_j = np.linalg.solve(
                gram + 1e-10 * np.eye(rank), (unfold_j @ khatri).T
            ).T

        approx = np.einsum("pr,ir,jr->pij", factor_p, factor_i, factor_j)
        best = min(best, float(np.linalg.norm(approx - target) / norm))

    return best


def main() -> None:
    for width in (6, 8, 12):
        target = multiplication_tensor(width)
        print(f"\n=== multiplication tensor, W = {width} (2W-1 = {2 * width - 1}) ===")
        print(f"{'rank R':>7}  {'rel. Frobenius error':>21}")
        for rank in range(1, 2 * width + 2):
            error = als_cp(target, rank)
            flag = "  <-- exact" if error < 1e-8 else ""
            print(f"{rank:>7}  {error:>21.3e}{flag}")


if __name__ == "__main__":
    main()
