import numpy as np
from typing import Tuple, Dict, List, Optional, Union, Any


def compute_var_cvar(distribution: np.array, alpha: float = 0.95) -> Tuple[float, float]:
    """
    Computes the Value at Risk (VaR) and Conditional Value at Risk (CVaR) for a given distribution.

    Parameters:
    - distribution: A 1D array of values representing the distribution (e.g., returns, losses, revenues)
    - alpha: The confidence level (default is 0.95, meaning we are focusing on the (1-0.95) worst percentile)

    Returns:
    - var: The Value at Risk (VaR) at the specified alpha percentile
    - cvar: The Conditional Value at Risk (CVaR), which is the average of values beyond the VaR
    """
    sorted_dist = np.sort(distribution)
    var = np.percentile(sorted_dist, (1 - alpha) * 100)
    cvar = sorted_dist[sorted_dist <= var].mean()
    return var, cvar


def percentile_mean(data: np.array, low_percentile: float = 0, high_percentile: float = 100) -> float:
    """
    Computes the mean of the data within the specified percentile range.

    Parameters:
    - data: A 1D array of numerical values
    - low_percentile: The lower percentile for filtering data (default is 0)
    - high_percentile: The upper percentile for filtering data (default is 100)

    Returns:
    - The mean of the data within the specified percentile range
    """
    lower_bound = np.percentile(data, low_percentile)
    upper_bound = np.percentile(data, high_percentile)
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    return filtered_data.mean()
