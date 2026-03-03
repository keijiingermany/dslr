import math
import random
from concurrent.futures import ProcessPoolExecutor
from .utils import sigmoid, dot


# ── Batch Gradient Descent (mandatory) ────────

def train_binary_gd(X, y, lr=0.1, iters=5000):
    m, n = len(X), len(X[0])
    th = [0.0] * n
    lr_m = lr / m
    _e = math.exp
    for _ in range(iters):
        g = [0.0] * n
        for i in range(m):
            xi = X[i]
            z = sum(th[j] * xi[j] for j in range(n))
            h = (1.0 / (1.0 + _e(-z))
                 if z >= 0 else _e(z) / (1.0 + _e(z)))
            err = h - y[i]
            for j in range(n):
                g[j] += err * xi[j]
        for j in range(n):
            th[j] -= lr_m * g[j]
    return th


# ── Stochastic Gradient Descent (bonus) ──────

def train_binary_sgd(X, y, lr=0.05, epochs=500):
    m, n = len(X), len(X[0])
    th = [0.0] * n
    _e = math.exp
    rng = random.Random(42)
    idx = list(range(m))
    for _ in range(epochs):
        rng.shuffle(idx)
        for i in idx:
            xi = X[i]
            z = sum(th[j] * xi[j] for j in range(n))
            h = (1.0 / (1.0 + _e(-z))
                 if z >= 0 else _e(z) / (1.0 + _e(z)))
            err = h - y[i]
            for j in range(n):
                th[j] -= lr * err * xi[j]
    return th


# ── Mini-batch Gradient Descent (bonus) ──────

def train_binary_minibatch(
    X, y, lr=0.1, epochs=500, batch_size=32,
):
    m, n = len(X), len(X[0])
    th = [0.0] * n
    _e = math.exp
    rng = random.Random(42)
    idx = list(range(m))
    for _ in range(epochs):
        rng.shuffle(idx)
        for s in range(0, m, batch_size):
            batch = idx[s:s + batch_size]
            bs = len(batch)
            lr_bs = lr / bs
            g = [0.0] * n
            for i in batch:
                xi = X[i]
                z = sum(th[j] * xi[j] for j in range(n))
                h = (1.0 / (1.0 + _e(-z))
                     if z >= 0
                     else _e(z) / (1.0 + _e(z)))
                err = h - y[i]
                for j in range(n):
                    g[j] += err * xi[j]
            for j in range(n):
                th[j] -= lr_bs * g[j]
    return th


# ── Dispatcher / OvR training ────────────────

OPTIMIZERS = {"batch", "sgd", "minibatch"}


def _train_one_class(args):
    """Module-level for ProcessPoolExecutor pickling."""
    X, y_bin, opt, kw = args
    if opt == "sgd":
        return train_binary_sgd(X, y_bin, **kw)
    if opt == "minibatch":
        return train_binary_minibatch(X, y_bin, **kw)
    return train_binary_gd(X, y_bin, **kw)


def train_ovr(
    X, y, classes, lr=0.1, iters=5000,
    optimizer="batch", batch_size=32,
):
    if optimizer == "sgd":
        kw = {"lr": lr, "epochs": iters}
    elif optimizer == "minibatch":
        kw = {"lr": lr, "epochs": iters,
              "batch_size": batch_size}
    else:
        kw = {"lr": lr, "iters": iters}

    tasks = [
        (X, [1 if yi == c else 0 for yi in y],
         optimizer, kw)
        for c in classes
    ]
    with ProcessPoolExecutor(len(classes)) as pool:
        res = list(pool.map(_train_one_class, tasks))
    return {c: th for c, th in zip(classes, res)}


def predict_ovr_one(x, thetas):
    best_c, best_p = None, -1.0
    for c, th in thetas.items():
        p = sigmoid(dot(th, x))
        if p > best_p:
            best_p, best_c = p, c
    return best_c
