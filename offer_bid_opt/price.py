import numpy as np
from typing import Tuple, Dict, List, Optional, Union, Any


def expected_value_at_p(a: np.array, b: np.array, p: float) -> float:
    """
    Computes the expected value of (a - b) * I(a >= p).

    Parameters:
    - a: Array of 'a' values (e.g., dalmp)
    - b: Array of 'b' values (e.g., rtlmp)
    - p: Threshold price 'p' to compute the expected value for

    Returns:
    - The expected value, or 0 if no valid elements exist (i.e., a >= p).
    """
    mask = a >= p
    total = np.sum((a[mask] - b[mask]))
    count = len(a)
    return total / count if count > 0 else 0  # Return 0 if no valid elements


def expected_value_at_p_reverse(a: np.array, b: np.array, p: float) -> float:
    """
    Computes the expected value of (b - a) * I(a >= p).

    Parameters:
    - a: Array of 'a' values (e.g., dalmp)
    - b: Array of 'b' values (e.g., rtlmp)
    - p: Threshold price 'p' to compute the expected value for

    Returns:
    - The expected value, or 0 if no valid elements exist (i.e., a >= p).
    """
    mask = a >= p
    total = np.sum((b[mask] - a[mask]))
    count = len(a)
    return total / count if count > 0 else 0  # Return 0 if no valid elements


def find_p_mean_rtlmp(data: np.array) -> Tuple[Dict[str, float], List[float]]:
    """
    Computes the optimal price as the mean of rtlmp and the expected value at each price.

    Parameters:
    - data: 3D array containing data for dalmp and rtlmp for each hour

    Returns:
    - p_opt: A dictionary of optimal prices for each time step
    - e_opt: A list of expected values corresponding to each time step
    """
    _, num_hours, _ = data.shape
    p_opt = {
        t: float(data[:, t, 1].mean())
        for t in range(num_hours)
    }
    e_opt = [expected_value_at_p(data[:, t, 0], data[:, t, 1], price) for t, price in p_opt.items()]
    return p_opt, e_opt


def helper_find_argmax_p(a: np.array, b: np.array, lo_hi: Tuple[float, float] = (0, 50)) -> Tuple[float, float]:
    """
    Helper function to find the price 'p' that maximizes the expected value.
    (a-b)(a > p)

    Parameters:
    - a: Array of 'a' values (e.g., dalmp)
    - b: Array of 'b' values (e.g., rtlmp)
    - lo_hi: Tuple specifying the lower and upper bounds for the search

    Returns:
    - max_val: The maximum expected value
    - max_p: The price 'p' that maximizes the expected value
    """
    lo, hi = lo_hi
    a = np.array(a)
    b = np.array(b)

    # Initialize max_val and max_p
    max_val = -float('inf')
    max_p = None

    # Search for the optimal price p
    for i in range(0, int((hi - lo) * 30)):
        p = lo + i / 30
        ev = expected_value_at_p(a, b, p)
        if ev > max_val:
            max_val = ev
            max_p = p

    return max_val, max_p


def find_p_argmax(data: np.array, negative: bool = False) -> Tuple[Dict[str, float], List[float]]:
    """
    Computes the optimal price 'p' by maximizing the expected value.

    Parameters:
    - data: 3D array containing data for dalmp and rtlmp for each hour
    - negative: Boolean flag to adjust the search range (used to handle negative prices)

    Returns:
    - p_opt: A dictionary of optimal prices for each time step
    - e_opt: A list of expected values corresponding to each time step
    """
    _, num_hours, _ = data.shape
    p_opt = {}
    e_opt = []

    for t in range(num_hours):
        dalmp = data[:, t, 0]
        rtlmp = data[:, t, 1]
        lo_hi = (
            int(dalmp.min()) - 1 if negative else 0,
            int(dalmp.max()) + 1,
        )
        max_val, max_p = helper_find_argmax_p(list(dalmp), list(rtlmp), lo_hi)
        p_opt[t] = max_p
        e_opt.append(max_val)

    return p_opt, e_opt


