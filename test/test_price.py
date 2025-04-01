import pytest
import numpy as np
from offer_bid_opt.price import expected_value_at_p, find_p_mean_rtlmp, find_p_argmax, helper_find_argmax_p


# Test for `expected_value_at_p`
def test_expected_value_at_p():
    a = np.array([50, 60, 70, 80])  # Example dalmp
    b = np.array([30, 40, 50, 60])  # Example rtlmp
    p = 65  # Threshold price

    result = expected_value_at_p(a, b, p)
    expected_result = (70 - 50) * 1 + (80 - 60) * 1  # (a - b) * I(a >= p)
    expected_result /= len(a)  # Divide by number of elements

    assert result == expected_result, f"Expected {expected_result}, but got {result}"


# Test for `find_p_mean_rtlmp`
def test_find_p_mean_rtlmp():
    # Sample data (samples x hours x variables)
    data = np.array([[[50, 30, 100],
                      [60, 35, 80]],  # Scenario 1
                     [[55, 32, 90],
                      [65, 38, 85]]])  # Scenario 2

    p_opt, e_opt = find_p_mean_rtlmp(data)

    # Check if the optimal price is calculated as the mean of rtlmp for each hour
    assert np.isclose(p_opt[0], np.mean(data[:, 0, 1])), f"Expected p_opt[0] to be {np.mean(data[:, 0, 1])}, but got {p_opt[0]}"
    assert np.isclose(p_opt[1], np.mean(data[:, 1, 1])), f"Expected p_opt[1] to be {np.mean(data[:, 1, 1])}, but got {p_opt[1]}"
    assert len(e_opt) == 2, "Expected 2 expected values (one for each hour)"
    assert e_opt[0] == expected_value_at_p(data[:,0,0],data[:,0,1],p_opt[0])
    assert e_opt[1] == expected_value_at_p(data[:,1,0],data[:,1,1],p_opt[1])

# Test for `helper_find_argmax_p`
def test_helper_find_argmax_p():
    a = np.array([50, 60, 70, 80])  # Example dalmp
    b = np.array([30, 80, 100, 20])  # Example rtlmp
    lo_hi = (0, 100)  # Search range for p

    max_val, max_p = helper_find_argmax_p(a, b, lo_hi)

    # Check if the max_p and max_val returned are reasonable (based on the example)
    assert max_p is not None, "Expected a valid price to be returned"
    assert max_val == 15
    assert max_p == pytest.approx(70.03333, rel=1e-1), "Expected a non-negative expected value"
    assert max_p <= a.max(), "Expected max_p to be within the given range"



# Test for `find_p_argmax`
def test_find_p_argmax():
    data = np.array([[[50, 30, 100],
                      [60, 35, 80]],  # Scenario 1
                     [[55, 32, 90],
                      [65, 38, 85]]])  # Scenario 2

    p_opt, e_opt = find_p_argmax(data, negative=False)

    # Check if the optimal price is within a reasonable range
    assert len(p_opt) == 2, "Expected 2 optimal prices (one for each hour)"
    assert len(e_opt) == 2, "Expected 2 expected values (one for each hour)"

    # Test if the optimal prices are reasonable
    for t in range(2):
        assert p_opt[t] >= 0, f"Expected p_opt[{t}] to be non-negative, but got {p_opt[t]}"
        assert e_opt[t] >= 0, f"Expected e_opt[{t}] to be non-negative, but got {e_opt[t]}"


# Edge case tests
def test_expected_value_at_p_edge_case():
    a = np.array([50, 60, 70, 80])
    b = np.array([30, 40, 50, 60])
    p = 100  # No values greater than or equal to 100

    result = expected_value_at_p(a, b, p)
    assert result == 0, f"Expected result to be 0, but got {result}"


def test_find_p_mean_rtlmp_no_valid_elements():
    data = np.array([[[10, 30, 100],
                      [20, 35, 80]],  # Scenario 1
                     [[15, 32, 90],
                      [25, 38, 85]]])  # Scenario 2
    
    p_opt, e_opt = find_p_mean_rtlmp(data)
    
    # Check if the optimal prices and expected values are reasonable
    for t in range(2):
        assert np.isclose(p_opt[t], np.mean(data[:, t, 1])), f"Expected p_opt[{t}] to be {np.mean(data[:, t, 1])}, but got {p_opt[t]}"
        assert e_opt[t] == 0, f"Expected e_opt[{t}] to be 0 when no elements are greater than or equal to p"

