import math
from typing import Dict, List
from .utils import sigmoid, dot


def train_binary_gd(
    X: List[List[float]],
    y: List[int],
    lr: float = 0.1,
    iters: int = 5000,
) -> List[float]:
    """
    Batch gradient descent for logistic regression (binary).
    X already includes bias term.
    """
    m = len(X)
    n = len(X[0])
    theta = [0.0] * n
    lr_m = lr / m

    # Cache math.exp as a local variable so the hot loop uses LOAD_FAST
    _exp = math.exp

    for _ in range(iters):
        grad = [0.0] * n
        for i in range(m):
            xi = X[i]

            # dot: explicit for-loop (faster than a sum() generator)
            z = 0.0
            for j in range(n):
                z += theta[j] * xi[j]
            if z >= 0.0:
                h = 1.0 / (1.0 + _exp(-z))
            else:
                e = _exp(z)
                h = e / (1.0 + e)

            err = h - y[i]
            for j in range(n):
                grad[j] += err * xi[j]

        for j in range(n):
            theta[j] -= lr_m * grad[j]

    return theta


def predict_proba_binary(theta: List[float], x: List[float]) -> float:
    return sigmoid(dot(theta, x))


def _train_one_class(args):
    """
    Top-level function required by ProcessPoolExecutor.

    It must be picklable (module-level) so it can be dispatched to worker
    processes.
    """
    X, y_bin, lr, iters = args
    return train_binary_gd(X, y_bin, lr=lr, iters=iters)


def train_ovr(
    X: List[List[float]],
    y_labels: List[str],
    classes: List[str],
    lr: float = 0.1,
    iters: int = 5000,
) -> Dict[str, List[float]]:
    """
    One-vs-Rest training.
    Each class' binary classifier is independent, so we parallelize training
    across classes using ProcessPoolExecutor. We use processes (not threads)
    to avoid the Global Interpreter Lock and achieve true parallelism.
    """
    from concurrent.futures import ProcessPoolExecutor

    tasks = [
        (X, [1 if yy == c else 0 for yy in y_labels], lr, iters)
        for c in classes
    ]

    with ProcessPoolExecutor(max_workers=len(classes)) as pool:
        results = list(pool.map(_train_one_class, tasks))

    return {c: th for c, th in zip(classes, results)}


def predict_ovr_one(x: List[float], thetas: Dict[str, List[float]]) -> str:
    best_c = None
    best_p = -1.0
    for c, th in thetas.items():
        p = predict_proba_binary(th, x)
        if p > best_p:
            best_p = p
            best_c = c
    return str(best_c)
