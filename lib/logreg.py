#!/usr/bin/env python3
from typing import Dict, List, Tuple
from .utils import sigmoid, dot


def train_binary_gd(
    X: List[List[float]],
    y: List[int],
    lr: float = 0.05,
    iters: int = 3000,
) -> List[float]:
    """
    Batch gradient descent for logistic regression (binary).
    X already includes bias term.
    """
    m = len(X)
    n = len(X[0])
    theta = [0.0] * n

    for _ in range(iters):
        grad = [0.0] * n
        for i in range(m):
            h = sigmoid(dot(theta, X[i]))
            err = h - y[i]
            xi = X[i]
            for j in range(n):
                grad[j] += err * xi[j]
        inv_m = 1.0 / m
        for j in range(n):
            theta[j] -= lr * grad[j] * inv_m

    return theta


def predict_proba_binary(theta: List[float], x: List[float]) -> float:
    return sigmoid(dot(theta, x))


def train_ovr(
    X: List[List[float]],
    y_labels: List[str],
    classes: List[str],
    lr: float = 0.05,
    iters: int = 3000,
) -> Dict[str, List[float]]:
    thetas: Dict[str, List[float]] = {}
    for c in classes:
        y_bin = [1 if yy == c else 0 for yy in y_labels]
        thetas[c] = train_binary_gd(X, y_bin, lr=lr, iters=iters)
    return thetas


def predict_ovr_one(x: List[float], thetas: Dict[str, List[float]]) -> str:
    best_c = None
    best_p = -1.0
    for c, th in thetas.items():
        p = predict_proba_binary(th, x)
        if p > best_p:
            best_p = p
            best_c = c
    return str(best_c)