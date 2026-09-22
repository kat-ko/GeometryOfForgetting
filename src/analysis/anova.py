"""Balanced two-way ANOVA with interaction, for the unique-n tests."""

from __future__ import annotations

import numpy as np
from scipy import stats


def two_way_balanced(data: np.ndarray) -> dict:
    """`data` is `(n_row, n_col, n_rep)`. Returns SS, df, F, p for both mains and the interaction."""
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 3:
        raise ValueError(f"expected (row, col, rep); got shape {data.shape}")
    a, b, n = data.shape
    if n < 2:
        raise ValueError("need at least 2 replicates per cell")
    grand = data.mean()
    row_m = data.mean(axis=(1, 2))
    col_m = data.mean(axis=(0, 2))
    cell_m = data.mean(axis=2)
    ss_row = b * n * np.sum((row_m - grand) ** 2)
    ss_col = a * n * np.sum((col_m - grand) ** 2)
    ss_ab = n * np.sum((cell_m - row_m[:, None] - col_m[None, :] + grand) ** 2)
    ss_err = np.sum((data - cell_m[:, :, None]) ** 2)
    df_row, df_col, df_ab = a - 1, b - 1, (a - 1) * (b - 1)
    df_err = a * b * (n - 1)
    ms_row, ms_col, ms_ab, ms_err = (
        ss_row / df_row, ss_col / df_col, ss_ab / df_ab, ss_err / df_err)
    F_row, F_col, F_ab = ms_row / ms_err, ms_col / ms_err, ms_ab / ms_err
    return {
        "n_row": a, "n_col": b, "n_rep": n,
        "ss_row": float(ss_row), "ss_col": float(ss_col), "ss_interaction": float(ss_ab),
        "ss_error": float(ss_err),
        "df_row": df_row, "df_col": df_col, "df_interaction": df_ab, "df_error": df_err,
        "F_row": float(F_row), "F_col": float(F_col), "F_interaction": float(F_ab),
        "p_row": float(stats.f.sf(F_row, df_row, df_err)),
        "p_col": float(stats.f.sf(F_col, df_col, df_err)),
        "p_interaction": float(stats.f.sf(F_ab, df_ab, df_err)),
    }


def two_way_interaction_ols(y, row, col, row_levels, col_levels) -> dict:
    """Type II F for the interaction in `y ~ row * col`, allowing unbalanced cells."""
    y = np.asarray(y, dtype=np.float64)
    row = np.asarray(row)
    col = np.asarray(col)

    def dummies(levels, values):
        cols = [(values == u).astype(np.float64) for u in levels[1:]]
        return np.column_stack(cols) if cols else np.zeros((len(values), 0))

    R = dummies(list(row_levels), row)
    C = dummies(list(col_levels), col)
    inter = [R[:, i] * C[:, j] for i in range(R.shape[1]) for j in range(C.shape[1])]
    I = np.column_stack(inter) if inter else np.zeros((len(y), 0))
    ones = np.ones((len(y), 1))
    X_full = np.column_stack([ones, R, C, I])
    X_red = np.column_stack([ones, R, C])

    def ss_res(X):
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        return float(resid @ resid), int(X.shape[1])

    ss_f, p_f = ss_res(X_full)
    ss_r, p_r = ss_res(X_red)
    df_int = p_f - p_r
    df_err = len(y) - p_f
    if df_int <= 0 or df_err <= 0 or ss_f <= 0:
        raise ValueError("interaction OLS is rank-deficient")
    F = ((ss_r - ss_f) / df_int) / (ss_f / df_err)
    return {
        "F_interaction": float(F),
        "p_interaction": float(stats.f.sf(F, df_int, df_err)),
        "df_interaction": int(df_int),
        "df_error": int(df_err),
        "n": int(len(y)),
    }


def fill_balanced(y, row, col, row_levels, col_levels) -> np.ndarray:
    """Stack unique-unit observations into `(n_row, n_col, n_rep)`. Requires equal n."""
    y = np.asarray(y, dtype=np.float64)
    row = np.asarray(row)
    col = np.asarray(col)
    cells = []
    ns = []
    for r in row_levels:
        for c in col_levels:
            mask = (row == r) & (col == c)
            cells.append(y[mask])
            ns.append(int(mask.sum()))
    if len(set(ns)) != 1:
        raise ValueError(f"unbalanced cells: n={ns}")
    n = ns[0]
    a, b = len(row_levels), len(col_levels)
    data = np.empty((a, b, n), dtype=np.float64)
    k = 0
    for i in range(a):
        for j in range(b):
            data[i, j] = cells[k]
            k += 1
    return data
