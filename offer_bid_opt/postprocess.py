import pyomo.environ as pyo
import numpy as np
from offer_bid_opt.utility import *
from offer_bid_opt.constants import * 
from typing import Tuple, Dict, List, Optional, Union, Any

def get_revenue_spread(m: pyo.ConcreteModel, strategy: str) -> np.array:
    """
    Retrieves the revenue spread for a given strategy.

    Parameters:
    - m: Pyomo ConcreteModel instance
    - strategy: A string indicating the strategy to use ('0' for SELF_SCHEDULE, '1' for ECONOMIC_BID_P_MEAN, '2' for ECONOMIC_BID_P_ARGMAX)

    Returns:
    - A NumPy array of revenue values corresponding to the strategy.
    """
    if strategy == SELF_SCHEDULE:
        # Return the revenue for each scenarios for SELF_SCHEDULE strategy
        return np.array([pyo.value(m.revenue[s]) for s in m.scenarios])
    elif strategy in [ECONOMIC_BID_P_MEAN, ECONOMIC_BID_P_ARGMAX, ECONOMIC_BID_DUAL]:
        # Return the revenue for each scenarios for ECONOMIC_BID_P_MEAN, ECONOMIC_BID_P_ARGMAX strategy
        return np.array([pyo.value(m.revenue_econ[s]) for s in m.scenarios])

def post_solve(m: pyo.ConcreteModel, strategy: str) -> Tuple[np.array, np.array]:
    """
    Post-solve function to retrieve bidding plan and revenue for a given strategy.

    Parameters:
    - m: Pyomo ConcreteModel instance
    - strategy: A string indicating the strategy to use ('0', '1', or '2')

    Returns:
    - A tuple of NumPy arrays:
        1. bidding_plan: A 2D array with price, offer, and bid. (number of hours x 3)
        2. revenue: A 1D array of revenue values. (number of scenarios)
    """
    # Generate offer and bid arrays based on the strategy
    if strategy == SELF_SCHEDULE:
        bid_offer = np.array([pyo.value(m.quantity_offer[t]) for t in m.times])
        offer = np.where(bid_offer >= 0, bid_offer, 0)
        bid = np.where(bid_offer < 0, -bid_offer, 0)
        price_offer = np.full(len(m.times), -1)
        price_bid = np.full(len(m.times), -1)
    elif strategy == ECONOMIC_BID_P_MEAN or strategy == ECONOMIC_BID_P_ARGMAX :
        offer = np.array([pyo.value(m.quantity_offer[t]) for t in m.times])
        bid = np.array([pyo.value(m.quantity_bid[t]) for t in m.times])
        price = np.array([pyo.value(m.price[t]) for t in m.times])
        price_offer = np.where(offer > 0, price, 0)
        price_bid = np.where(bid > 0, price, 0)
    elif strategy == ECONOMIC_BID_DUAL:
        offer = np.array([pyo.value(m.quantity_offer[t]) for t in m.times])
        bid = np.array([pyo.value(m.quantity_bid[t]) for t in m.times])
        price_offer = np.array([pyo.value(m.price_offer[t]) for t in m.times])
        price_bid = np.array([pyo.value(m.price_bid[t]) for t in m.times])

    # Stack price, offer, and bid to create the bidding plan
    bidding_plan = np.column_stack((price_offer, price_bid, offer, bid))

    # Get revenue spread based on the strategy
    revenue = get_revenue_spread(m, strategy)

    return bidding_plan, revenue
