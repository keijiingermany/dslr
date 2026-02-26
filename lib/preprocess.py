import math
from typing import List, Tuple


def column_means(X: List[List[float]]) -> List[float]:
    m = len(X)
    n = len(X[0])
    means = []
    for j in range(n):
        s = 0.0
        c = 0
        for i in range(m):
            v = X[i][j]
            if not math.isnan(v):
                s += v
                c += 1
        means.append(s / c if c > 0 else 0.0)
    return means


def fill_nan_with_means(X: List[List[float]], means: List[float]) -> None:
    m = len(X)
    n = len(X[0])
    for i in range(m):
        for j in range(n):
            if math.isnan(X[i][j]):
                X[i][j] = means[j]


def column_stds_sample(X: List[List[float]], means: List[float]) -> List[float]:
    """
    sample std over rows for each column; if std=0 -> set to 1 to avoid div-by-zero.
    Assumes no NaN (fill first).
    """
    m = len(X)
    n = len(X[0])
    stds = []
    for j in range(n):
        if m <= 1:
            stds.append(1.0)
            continue
        var = 0.0
        mu = means[j]
        for i in range(m):
            d = X[i][j] - mu
            var += d * d
        var /= (m - 1)
        sd = math.sqrt(var)
        stds.append(sd if sd > 0 else 1.0)
    return stds


def standardize_inplace(X: List[List[float]], means: List[float], stds: List[float]) -> None:
    m = len(X)
    n = len(X[0])
    for i in range(m):
        for j in range(n):
            X[i][j] = (X[i][j] - means[j]) / stds[j]


def add_bias(X: List[List[float]]) -> List[List[float]]:
    return [[1.0] + row[:] for row in X]


def preprocess_fit_transform(X: List[List[float]]) -> Tuple[List[List[float]], List[float], List[float]]:
    """
    Fit on X: means/stds, fill NaN, standardize, add bias.
    Returns (X_processed, means, stds)
    """
    means = column_means(X)
    fill_nan_with_means(X, means)
    stds = column_stds_sample(X, means)
    standardize_inplace(X, means, stds)
    Xb = add_bias(X)
    return Xb, means, stds


def preprocess_transform(X: List[List[float]], means: List[float], stds: List[float]) -> List[List[float]]:
    """
    Transform with given means/stds, fill NaN with means, standardize, add bias.
    """
    fill_nan_with_means(X, means)
    standardize_inplace(X, means, stds)
    return add_bias(X)