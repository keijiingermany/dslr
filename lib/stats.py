import math


def mean(data):
    return sum(data) / len(data)


def var_sample(data, mu):
    n = len(data)
    if n <= 1:
        return 0.0
    return sum((x - mu) ** 2 for x in data) / (n - 1)


def std_sample(data, mu):
    return math.sqrt(var_sample(data, mu))


def compute_min(data):
    m = data[0]
    for x in data[1:]:
        if x < m:
            m = x
    return m


def compute_max(data):
    m = data[0]
    for x in data[1:]:
        if x > m:
            m = x
    return m


def compute_range(data):
    return compute_max(data) - compute_min(data)


def percentile_linear(sorted_data, p):
    n = len(sorted_data)
    if n == 1:
        return sorted_data[0]
    k = (n - 1) * p
    f, c = int(math.floor(k)), int(math.ceil(k))
    if f == c:
        return sorted_data[f]
    return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)


def iqr(sorted_data):
    return (percentile_linear(sorted_data, 0.75)
            - percentile_linear(sorted_data, 0.25))


def skewness(data, mu):
    n = len(data)
    if n < 3:
        return 0.0
    sd = std_sample(data, mu)
    if sd == 0.0:
        return 0.0
    m3 = sum((x - mu) ** 3 for x in data) / n
    adj = (n * (n + 1)) / ((n - 1) * (n - 2))
    return adj * m3 / (sd ** 3)


def kurtosis_excess(data, mu):
    n = len(data)
    if n < 4:
        return 0.0
    sd = std_sample(data, mu)
    if sd == 0.0:
        return 0.0
    s4 = sum(((x - mu) / sd) ** 4 for x in data)
    t1 = (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))
    t2 = 3.0 * ((n - 1) ** 2) / ((n - 2) * (n - 3))
    return t1 * s4 - t2


def count_unique(data):
    return len(set(data))
