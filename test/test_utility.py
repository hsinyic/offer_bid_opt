import numpy as np
import pytest
from offer_bid_opt.utility import compute_var_cvar, percentile_mean


def test_compute_var_cvar_uniform():
    # Test case with a normal distribution
    data = np.random.uniform(0, 1, 10000)  # Generate 10000 random values from a normal distribution
    var, cvar = compute_var_cvar(data, alpha=0.95)

    
    # Check if the VaR is close to 0.05 (the expected value for 95% confidence level in a uniform [0, 1] distribution)
    assert np.isclose(var, 0.05, atol=0.01), f"VaR {var} is not close to 0.05, expected value."

    # Check if CVaR is close to 0.025 (the average of the worst 5% of a uniform [0, 1] distribution)
    assert np.isclose(cvar, 0.025, atol=0.01), f"CVaR {cvar} is not close to 0.025, expected value."


# Test for normal distribution of data
def test_compute_var_cvar_normal():
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11])  # Simple example data
    var, cvar = compute_var_cvar(data, alpha=0.9)

    # VaR should be within the data range, and CVaR should be <= VaR
    assert var == pytest.approx(2, 1e-1), f"Expected VaR to be 2, but got {var}"
    assert cvar == 1, f"Expected CVaR to be 1, but got {cvar}"


# Test for edge case: data with identical values
def test_compute_var_cvar_identical():
    data = np.array([5, 5, 5, 5, 5])  # All values are the same
    var, cvar = compute_var_cvar(data, alpha=0.95)

    # Both VaR and CVaR should be 5 because all values are the same
    assert var == 5, f"Expected VaR to be 5, but got {var}"
    assert cvar == 5, f"Expected CVaR to be 5, but got {cvar}"

# Test for normal data
def test_percentile_mean_normal():
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    mean = percentile_mean(data, 0, 100)  # Mean of the entire data
    assert mean == 5.5, f"Expected mean to be 5.5, but got {mean}"

# Test for custom percentile range (50-90 percentile)
def test_percentile_mean_range():
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    mean = percentile_mean(data, 50, 90)
    assert mean == 7.5, f"Expected mean to be 7.5, but got {mean}"
