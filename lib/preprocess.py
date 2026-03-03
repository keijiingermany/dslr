import math


def preprocess_fit_transform(X):
    m, n = len(X), len(X[0])
    # column means (ignoring NaN)
    means = []
    for j in range(n):
        vals = [X[i][j] for i in range(m) if not math.isnan(X[i][j])]
        means.append(sum(vals) / len(vals) if vals else 0.0)
    # fill NaN
    for i in range(m):
        for j in range(n):
            if math.isnan(X[i][j]):
                X[i][j] = means[j]
    # sample stds
    stds = []
    for j in range(n):
        if m <= 1:
            stds.append(1.0)
            continue
        var = sum((X[i][j] - means[j]) ** 2 for i in range(m)) / (m - 1)
        stds.append(math.sqrt(var) or 1.0)
    # standardize + add bias
    Xb = [[1.0] + [(X[i][j] - means[j]) / stds[j] for j in range(n)]
          for i in range(m)]
    return Xb, means, stds


def preprocess_transform(X, means, stds):
    m, n = len(X), len(X[0])
    for i in range(m):
        for j in range(n):
            if math.isnan(X[i][j]):
                X[i][j] = means[j]
    return [[1.0] + [(X[i][j] - means[j]) / stds[j] for j in range(n)]
            for i in range(m)]
