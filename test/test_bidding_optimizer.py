import pytest
import pyomo.environ as pyo
import numpy as np
from offer_bid_opt.bidding_optimizer import BiddingOptimizer
from offer_bid_opt.constants import (
    SELF_SCHEDULE,
    ECONOMIC_BID_P_MEAN,
    ECONOMIC_BID_P_ARGMAX,
)
from offer_bid_opt.price import find_p_argmax
from offer_bid_opt.sample_data_generator import *
from offer_bid_opt.constants import *
from offer_bid_opt.utility import *

# TODO: ensure glpk is installed 
solver_option = ['glpk', None, False]

def test_self_schedule():
    # Define a simple dataset: [DA LMP, RT LMP, Wind Generation]
    test_data = np.array(
        [[[50.0, 30.0, 100.0], [50.0, 30.0, 70.0]]]  # Hour 0  # Hour 1
    )

    # Initialize optimizer and set test data
    optimizer = BiddingOptimizer()
    optimizer.set_wind_capacity_mw(120)
    optimizer.set_data(test_data)
    optimizer.set_strategy(SELF_SCHEDULE)

    # Run optimization
    optimizer.optimize(solver_option=solver_option)

    # Retrieve bidding plan
    bidding_plan = optimizer.get_bidding_plan()

    # Expected offer quantities (should match wind generation)
    expected_bidding_plan = np.array(
        [
            [-1, -1, 100, 0],
            [-1, -1, 70, 0],
        ]
    )

    # Ensure that the bid offer matches the expected wind generation
    assert np.allclose(
        bidding_plan, expected_bidding_plan
    ), f"Expected {expected_bidding_plan}, but got {bidding_plan}"

    # Run optimization with CVar values, result should be exactly the same due to there being 1 scenario
    optimizer.optimize(cvar={"alpha": 0.95, "beta": 0.5}, solver_option=solver_option)
    bidding_plan = optimizer.get_bidding_plan()
    assert np.allclose(
        bidding_plan, expected_bidding_plan
    ), f"Expected {expected_bidding_plan}, but got {bidding_plan}"


def test_econ_bid_single_scenario():
    # Define a simple dataset: [DA LMP, RT LMP, Wind Generation]
    test_data = np.array(
        [
            [
                [50.0, 30.0, 100.0],  # Hour 0
                [50.0, 30.0, 70.0],  # Hour 1
            ]
        ]
    )

    # Initialize optimizer and set test data
    optimizer = BiddingOptimizer()
    optimizer.set_wind_capacity_mw(120)
    optimizer.set_data(test_data)
    optimizer.set_strategy(ECONOMIC_BID_P_MEAN)

    # Run optimization
    optimizer.optimize(solver_option=solver_option)

    # Retrieve bidding plan
    bidding_plan = optimizer.get_bidding_plan()

    # Expected offer quantities (should match wind generation)
    price_offer = np.array([30, 30])  # mean RTLMP of each hour
    price_bid = np.array([0, 0])  # mean RTLMP of each hour
    offer = np.array([100, 70])  # offering at maximum allowed to capture arbitrage
    bid = np.array([0, 0])
    expected_bidding_plan = np.column_stack((price_offer, price_bid, offer, bid))

    # Ensure that the bid offer matches the expected wind generation
    assert np.allclose(
        bidding_plan, expected_bidding_plan
    ), f"Expected {expected_bidding_plan}, but got {bidding_plan}"


def test_econ_bid_multiple_scenarios():
    # Define a simple dataset: [DA LMP, RT LMP, Wind Generation]
    test_data = np.array(
        [[[80, 80, 100], [30, 70, 80]], [[55, 10, 90], [65, 50, 85]]]  # Scenario 1
    )  # Scenario 2

    # Initialize optimizer and set test data
    optimizer = BiddingOptimizer()
    optimizer.set_wind_capacity_mw(120)
    optimizer.set_data(test_data)
    optimizer.set_strategy(
        ECONOMIC_BID_P_ARGMAX
    )  # test that if we change strategy, nothing would break
    optimizer.set_strategy(ECONOMIC_BID_P_MEAN)

    # Run optimization
    optimizer.optimize(solver_option=solver_option)

    # Retrieve bidding plan
    bidding_plan = optimizer.get_bidding_plan()

    price = np.mean(test_data[:, :, 1], axis=0)  # mean RTLMP of each hour
    offer = np.array([95, 0])
    bid = np.array([0, 82.5])
    price_offer = np.where(offer > 0, price, 0 )  # mean RTLMP of each hour
    price_bid = np.where(bid > 0, price, 0 )  # mean RTLMP of each hour
    expected_bidding_plan = np.column_stack((price_offer, price_bid, offer, bid))

    assert np.allclose(
        bidding_plan, expected_bidding_plan
    ), f"Expected {expected_bidding_plan}, but got {bidding_plan}"

    optimizer.set_strategy(ECONOMIC_BID_P_ARGMAX)
    optimizer.optimize(solver_option=solver_option)
    bidding_plan = optimizer.get_bidding_plan()

    p_dict, _ = find_p_argmax(
        test_data
    )  # p is the argmax of an expected [(DA-RT price) * (DA >= p*)]
    price = list(p_dict.values())
    offer = np.array([95, 0])
    bid = np.array([0, 82.5])
    price_offer = np.where(offer > 0, price, 0 )  # mean RTLMP of each hour
    price_bid = np.where(bid > 0, price, 0 )  # mean RTLMP of each hour
    expected_bidding_plan = np.column_stack((price_offer, price_bid, offer, bid))
    assert np.allclose(
        bidding_plan, expected_bidding_plan
    ), f"Expected {expected_bidding_plan}, but got {bidding_plan}"


@pytest.mark.parametrize(
    "strategy",
    [
        SELF_SCHEDULE,
        ECONOMIC_BID_P_MEAN,
        ECONOMIC_BID_P_ARGMAX,
    ],
)
def test_econ_bid_cvar_var(strategy):
    test_data, _ = generate_sample_data(num_samples=10000, num_hours=3, wind_capacity_mw=150)
    optimizer = BiddingOptimizer()
    optimizer.set_wind_capacity_mw(150)
    optimizer.set_data(test_data)

    optimizer.set_strategy(
        strategy
    )  # test that if even as we change strategy, Cvar would be calculated correctly

    
    optimizer.optimize(cvar={'alpha':0.95, 'beta':0.5}, solver_option=solver_option)
    rev = optimizer.get_revenue()
    expected_var, expected_cvar = compute_var_cvar(rev)

    m = optimizer.bidding_model.get_model()
    var, cvar = m.zeta.value, pyo.value(m.cvar) 
    
    assert expected_cvar/cvar == pytest.approx(1, rel=0.05), f"Expected Var {expected_cvar}, but got {cvar}"
    assert expected_var/var == pytest.approx(1, rel=0.05), f"Expected Var {expected_var}, but got {var}"



    