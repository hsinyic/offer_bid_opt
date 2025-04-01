from pyomo.environ import *
from pyomo.opt import SolverFactory
from offer_bid_opt.bidding_model import BiddingModel
from offer_bid_opt.utility import *
from offer_bid_opt.postprocess import post_solve
from offer_bid_opt.sample_data_generator import generate_sample_data
from offer_bid_opt.visualize import plot_hourly_offer_bid
from offer_bid_opt.constants import STRATEGY_NAMES
import numpy as np
from typing import Optional, Dict, List, Tuple

class BiddingOptimizer:
    """
    A class to generate offer/bidding strategy outcomes for a wind farm.
    """
    
    def __init__(self):
        """
        Initializes the BiddingOptimizer with default parameters and sample data.
        """
        self.wind_capacity_mw: float = 150  # Wind farm capacity in MW
        self.num_hours: int = 24  # Number of hours in the optimization period
        self.num_samples: int = 500  # Number of scenarios/samples
        self.data: np.ndarray = generate_sample_data(self.num_samples, self.num_hours, self.wind_capacity_mw)
        
        self.strategy: Optional[str] = None
        self.bidding_model: Optional[BiddingModel] = None
        
        # Results
        self.revenue: Optional[np.ndarray] = None
        self.revenue_dict: Dict[str, np.ndarray] = {}
        self.bidding_plan: Optional[Dict] = None
        self.bidding_plan_dict: Dict[str, Dict] = {}
    
    def set_wind_capacity_mw(self, new_capacity: float) -> None:
        """
        Updates the wind farm capacity.
        
        :param new_capacity: New wind capacity in MW.
        """
        self.wind_capacity_mw = new_capacity

    def set_data(self, data: np.ndarray) -> None:
        """
        Sets the bidding data and updates the number of samples and hours accordingly.
        
        :param data: A NumPy array containing the bidding data.
        """
        num_samples, num_hours, _ = data.shape
        self.num_samples, self.num_hours = num_samples, num_hours
        self.data = data
    
    def get_bidding_plan_dict(self) -> Dict[str, Dict]:
        return self.bidding_plan_dict
    
    def get_revenue_dict(self) -> Dict[str, np.ndarray]:
        return self.revenue_dict
    
    def get_bidding_plan(self) -> Optional[Dict]:
        return self.bidding_plan
    
    def get_revenue(self) -> Optional[np.ndarray]:
        return self.revenue

    def set_strategy(self, new_strat: str) -> None:
        """
        Sets the bidding strategy and initializes or updates the bidding model.
        
        :param new_strat: The new strategy to be set.
        """
        if self.strategy is None:
            self.bidding_model = BiddingModel(new_strat, self.data, self.wind_capacity_mw)
        elif self.strategy != new_strat:
            self.bidding_model = BiddingModel(new_strat, self.data, self.wind_capacity_mw)
        self.strategy = new_strat
    
    def simulate_revenue(self, new_data: np.ndarray, overwrite_data: bool = False) -> np.ndarray:
        """
        Simulates revenue using new data and optionally overwrites the existing data.
        
        :param new_data: A NumPy array of new data with the same shape as the original.
        :param overwrite_data: If True, the original data will be overwritten.
        :return: The revenue spread as a NumPy array.
        """
        assert new_data.shape == self.data.shape, "New data shape must match the existing data shape."
        
        self.bidding_model.update_data(new_data)
        revenue = get_revenue_spread(self.bidding_model.get_model(), self.strategy)
        
        if not overwrite_data:
            self.bidding_model.update_data(self.data)
        
        return revenue
    
    def optimize(self, cvar: Optional[Dict[str, float]] = None, visualize_option: bool = False, solver_option: Tuple[str, Optional[str], bool] = ('glpk', None, False)) -> None:
        """
        Runs the bidding optimization process.
        
        :param cvar: A dictionary containing alpha and beta values for CVaR constraints.
        :param visualize_option: If True, visualizes the results.
        :param solver_option: Tuple containing solver name, optional path to executable, and tee option.
        """
        strategy, stra_name = self.strategy, STRATEGY_NAMES[self.strategy]

        if cvar is None or cvar['beta'] == 0.0:
            cvar = {'alpha': 0.95, 'beta': 0.0}  # Default CVaR values
            run_name = f'{stra_name}_nocvar'
        else:
            run_name = f'{stra_name}_alpha{cvar["alpha"]*100:.0f}_beta{cvar["beta"]*100:.0f}'
        
        self.bidding_model.load_cvar_alpha_beta(cvar)
        
        solver_name, path_to_executable, tee = solver_option
        solver = SolverFactory(solver_name, executable=path_to_executable) if path_to_executable else SolverFactory(solver_name)
        solver.solve(self.bidding_model.model, tee=tee)
        
        bidding_plan, revenues = post_solve(self.bidding_model.model, strategy)
        self.bidding_plan_dict[run_name] = bidding_plan
        self.revenue_dict[run_name] = revenues
        self.bidding_plan = bidding_plan
        self.revenue = revenues
        
        print("Avg revenue:", f"{percentile_mean(revenues):.1f}")
        print(f"Avg revenue from worst {100-cvar['alpha']*100}%:", f"{percentile_mean(revenues, cvar['alpha']):.1f}")
        
        if visualize_option:
            plot_hourly_offer_bid(self.bidding_model.model, run_name, strategy).show()
