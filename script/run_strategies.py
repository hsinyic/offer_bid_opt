from offer_bid_opt.bidding_optimizer import *
from offer_bid_opt.visualize import * 
from offer_bid_opt.postprocess import * 
from offer_bid_opt.sample_data_generator import * 
from offer_bid_opt.constants import * 

import os



# Get the absolute path to the data directory
base_dir = os.path.dirname(os.path.abspath("__main__"))  # This will get the directory of your script
data_dir = os.path.join(base_dir, 'data')  # Going one level up and then to the 'data' folder


bidding_instance = BiddingOptimizer()

windfarm_mw, filePath = 150, os.path.join(data_dir, 'energy_market_samples.npz')
windfarm_mw, filePath = 150, os.path.join(data_dir, 'energy_market_samples_large.npz')
data, _ = load_sample_data(filePath)
bidding_instance.set_data(data) # dalmp, rtlmp,windgen 
bidding_instance.set_wind_capacity_mw(windfarm_mw) 

# should be replaced by set_input_data (wind_capacity, data arrray of N samples, M hours, 3 variables of dalmp, rtlmp, windgenpower)
# check that wind_capacity >= max(windgenpower)
data, _ = generate_sample_data(500, 24, 150)
bidding_instance.set_data(data) # dalmp, rtlmp,windgen 
bidding_instance.set_wind_capacity_mw(150) 


# print("----- self schedule -------")
bidding_instance.set_strategy(new_strat=SELF_SCHEDULE)
bidding_instance.optimize(visualize_option=True, cvar={'alpha':0.8, 'beta':0}) 
bidding_instance.optimize(visualize_option=True, cvar={'alpha':0.95, 'beta':0.5}) 
bidding_instance.optimize(visualize_option=True, cvar={'alpha':0.95, 'beta':0.8}) 



# print("----- economic bidding strategies mean -------")
bidding_instance.set_strategy(new_strat=ECONOMIC_BID_P_MEAN)
bidding_instance.optimize(visualize_option=True, cvar={'alpha':0.8, 'beta':0}) 
bidding_instance.optimize(visualize_option=True, cvar={'alpha':0.95, 'beta':0.5}) 
bidding_instance.optimize(visualize_option=True, cvar={'alpha':0.95, 'beta':0.8}) 

# print("----- economic bidding strategies argmax -------")
bidding_instance.set_strategy(new_strat=ECONOMIC_BID_P_ARGMAX)
bidding_instance.optimize(visualize_option=True, cvar={'alpha':0.8, 'beta':0}) 
bidding_instance.optimize(visualize_option=True, cvar={'alpha':0.95, 'beta':0.5}) 
bidding_instance.optimize(visualize_option=True, cvar={'alpha':0.95, 'beta':0.8})  


print("----- economic bidding dual offer and bid: submit both offer and bid each with different price -------")
bidding_instance.set_strategy(new_strat=ECONOMIC_BID_DUAL)
solver_option = ['scip', '/Users/hchen/miniforge3/envs/Tensorflow/bin/scip', True]
bidding_instance.optimize(visualize_option=True, cvar={'alpha':0.8, 'beta':0},solver_option=solver_option) 
bidding_instance.optimize(visualize_option=True, cvar={'alpha':0.95, 'beta':0.5}, solver_option=solver_option)
bidding_instance.optimize(visualize_option=True, cvar={'alpha':0.95, 'beta':0.8}, solver_option=solver_option)
revenues_spread = bidding_instance.get_revenue_dict() 
plot_histogram(revenues_spread)
plot_strategy_averages(revenues_spread)

