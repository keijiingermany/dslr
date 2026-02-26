import math
from typing import List


def mean(data: List[float]) -> float:
    return sum(data) / len(data)


def std_sample(data: List[float], mu: float) -> float:
    n = len(data)
    if n <= 1:
        return 0.0
    var = 0.0
    for x in data:
        d = x - mu
        var += d * d
    var /= (n - 1)
    return math.sqrt(var)


def compute_min(data: List[float]) -> float:
    m = data[0]
    for x in data[1:]:
        if x < m:
            m = x
    return m


def compute_max(data: List[float]) -> float:
    m = data[0]
    for x in data[1:]:
        if x > m:
            m = x
    return m


def percentile_linear(sorted_data: List[float], p: float) -> float:
    """
    Linear interpolation percentile with index = p*(n-1).
    sorted_data must be sorted ascending, non-empty.
    """
    n = len(sorted_data)
    if n == 1:
        return sorted_data[0]

    k = (n - 1) * p
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return sorted_data[f]
    return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)
