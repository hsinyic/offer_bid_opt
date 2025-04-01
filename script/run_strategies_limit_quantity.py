from offer_bid_opt.bidding_optimizer import *
from offer_bid_opt.visualize import * 
from offer_bid_opt.postprocess import * 
from offer_bid_opt.sample_data_generator import * 
from offer_bid_opt.constants import * 
import os



# Get the absolute path to the data directory
base_dir = os.path.dirname(os.path.abspath(__file__))  # This will get the directory of your script
data_dir = os.path.join(base_dir, '..', 'data')  # Going one level up and then to the 'data' folder


bidding_instance = BiddingOptimizer()

windfarm_mw, filePath = 150, os.path.join(data_dir, 'energy_market_samples.npz')
windfarm_mw, filePath = 150, os.path.join(data_dir, 'energy_market_samples_large.npz')
bidding_instance.set_data(data) # dalmp, rtlmp,windgen 
bidding_instance.set_wind_capacity_mw(windfarm_mw) 
bidding_instance.set_strategy(new_strat=ECONOMIC_BID_P_ARGMAX)
bidding_instance.optimize(visualize_option=True, cvar={'alpha':0.95, 'beta':0.5}) 

bidding_instance_raise_limit = BiddingOptimizer()
bidding_instance_raise_limit.set_data(data) # dalmp, rtlmp,windgen 
bidding_instance_raise_limit.set_wind_capacity_mw(windfarm_mw*2) 
bidding_instance_raise_limit.set_strategy(new_strat=ECONOMIC_BID_P_ARGMAX)
bidding_instance_raise_limit.optimize(visualize_option=True, cvar={'alpha':0.95, 'beta':0.2}) 
