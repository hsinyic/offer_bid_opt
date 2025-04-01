from pyomo.environ import *
from pyomo.opt import SolverFactory
from offer_bid_opt.bidding_model import * 
from offer_bid_opt.utility import * 
from offer_bid_opt.postprocess import * 
from offer_bid_opt.sample_data_generator import * 
from offer_bid_opt.visualize import * 
from offer_bid_opt.constants import * 

class BiddingOptimizer():

    def __init__(self):
        # TODO: num_hours and num_samples are inferred from data
        # TODO: Not the best practice to include generated sample data duing initiation
        self.wind_capacity_mw = 150  
        self.num_hours = 24
        self.num_samples = 500 
        self.data = generate_sample_data(self.num_samples, self.num_hours, self.wind_capacity_mw) 

        self.strategy = None 
        self.bidding_model = None 

        # Results 
        self.revenue = None
        self.revenue_dict = {}
        self.bidding_plan = None
        self.bidding_plan_dict = {}

    def set_wind_capacity_mw(self, new_capacity):
        self.wind_capacity_mw = new_capacity

    def set_data(self, data: np.array): 
        self.data = data
        num_samples, num_hours, _ = data.shape
        self.num_samples, self.num_hours = num_samples, num_hours

    def get_bidding_plan_dict(self):
        return self.bidding_plan_dict 

    def get_revenue_dict(self):
        return self.revenue_dict 

    def get_bidding_plan(self):
        return self.bidding_plan 

    def get_revenue(self):
        return self.revenue 

    def set_strategy(self, new_strat):
        if self.strategy == None: 
            self.bidding_model = BiddingModel(new_strat, self.data, self.wind_capacity_mw)
        elif self.strategy != new_strat: # change in strategy 
            # self.bidding_model.update_model(new_strat)
            self.bidding_model = BiddingModel(new_strat, self.data, self.wind_capacity_mw)
        elif self.strategy == new_strat: # do nothing 
            return 

        self.strategy = new_strat 

    def simulate_revenue(self, new_data, overwrite_data=False):
        assert new_data.shape == self.data.shape
        # temporarily replace with new data 
        self.bidding_model.update_data(new_data)
        # calculate new revenue spread 
        revenue = get_revenue_spread(self.bidding_model.get_model(), self.strategy)
        if not overwrite_data: 
            # change back to original data
            self.bidding_model.update_data(self.data)
        return revenue
            

    def optimize(self, cvar=None, visualize_option=False, solver_option=['glpk', None, False]):
        strategy, stra_name = self.strategy, STRATEGY_NAMES[self.strategy]

        if cvar is None or cvar['beta'] == 0.0:
            cvar = {'alpha': 0.95, 'beta': 0.0}  # Default CVaR values
            run_name = f'{stra_name}_nocvar'
        else:
            run_name = f'{stra_name}_alpha{cvar["alpha"]*100:.0f}_beta{cvar["beta"]*100:.0f}'
        self.bidding_model.load_cvar_alpha_beta(cvar)

        # TODO: hacky 
        # solver = SolverFactory('glpk')
        # solver = SolverFactory('scip', executable='/Users/hchen/miniforge3/envs/Tensorflow/bin/scip')
        solver_name, path_to_executable, tee = solver_option
        if path_to_executable is None: 
            solver = SolverFactory(solver_name)
        else:
            solver = SolverFactory(solver_name, executable=path_to_executable)
        solver.solve(self.bidding_model.model, tee=tee) 

        bidding_plan, revenues = post_solve(self.bidding_model.model, strategy)
        self.bidding_plan_dict[run_name] = bidding_plan
        self.revenue_dict[run_name] = revenues
        self.bidding_plan  = bidding_plan
        self.revenue = revenues

        print("avg revenue: ", f"{percentile_mean(revenues):.1f}")
        print(f"avg revenue from worst {100-cvar['alpha']*100}%: ", f"{percentile_mean(revenues, cvar['alpha']):.1f}")

        # visualize hourly operations 
        if visualize_option:
            plot_hourly_offer_bid(self.bidding_model.model, run_name, strategy).show()
            


