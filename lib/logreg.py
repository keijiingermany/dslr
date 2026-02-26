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

    Speed-ups (pure Python, no external libs):
    1. jループを1回に統合: dot() + grad更新を1つのjループで処理。
       元の実装は dot() 内部で j を1回、grad更新で j をもう1回回していた。
    2. ローカル変数ホイスティング: math.exp をローカルにキャッシュし
       ホットループ内の LOAD_GLOBAL コストを LOAD_FAST に下げる。
    3. lr/m を定数として事前計算してループ外に出す。
    """
    m = len(X)
    n = len(X[0])
    theta = [0.0] * n
    lr_m = lr / m

    # ローカルにキャッシュ → ホットループ内が LOAD_FAST になる
    _exp = math.exp

    for _ in range(iters):
        grad = [0.0] * n
        for i in range(m):
            xi = X[i]

            # dot: 明示forループ（sum()ジェネレータより速い）
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
    """Top-level function required by ProcessPoolExecutor (must be picklable)."""
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
    各クラスの二値分類は独立しているので ProcessPoolExecutor で並列実行する。
    GIL の制約を回避するためスレッドではなくプロセスを使う。
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