import numpy as np


def ema(series: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Calculate Exponential Moving Average (EMA) for a given series

    Args:
        series: Input numerical sequence to smooth
        alpha: Smoothing factor (0 < alpha <= 1), higher values give more weight to recent values

    Returns:
        Smoothed EMA sequence with the same shape as input
    """
    smoothed = np.zeros_like(series, dtype=float)
    smoothed[0] = series[0]

    for i in range(1, len(series)):
        smoothed[i] = alpha * series[i] + (1 - alpha) * smoothed[i - 1]

    return smoothed
