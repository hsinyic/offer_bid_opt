import pyomo.environ as pyo
import numpy as np
from offer_bid_opt.utility import * 
from offer_bid_opt.price import * 
from offer_bid_opt.constants import * 
from typing import Tuple, Dict, List, Optional, Union, Any

class BiddingModel():

    def __init__(self, strategy:str, data:np.array, wind_capacity_mw:float):
        self.strategy = strategy 
        self.data = data
        self.wind_capacity_mw = wind_capacity_mw
        self.model = None
        self.build_model()

    def get_model(self) -> pyo.ConcreteModel:
        return self.model 
    
    def set_strategy(self, stra: str):
        self.strategy = stra 

    def build_model(self): # build model from scratch 
        if self.strategy == SELF_SCHEDULE:
            self.model = self.build_self_schedule_model()
        elif self.strategy == ECONOMIC_BID_DUAL:
            self.model = self.build_dual_econ_bid_model()
        else: 
            m = self.build_self_schedule_model()
            self.model = self.build_econ_bid_model(m)
            self.update_price()

    def expand_model(self):
        self.model = self.build_econ_bid_model(self.model)
        self.update_price()

    def build_self_schedule_model(self) -> pyo.ConcreteModel:
        data, wind_capacity_mw = self.data, self.wind_capacity_mw
        num_samples, num_hours, _ = data.shape
        m = pyo.ConcreteModel()

        # Sets 
        m.times = pyo.Set(initialize = [i for i in range(num_hours)], ordered=True)
        m.scenarios = pyo.Set(initialize = [i for i in range(num_samples)], ordered=False)

        # TODO: research methods to load param values faster 
        def param_rtlmp(m, t, s):
            return data[s][t][1]
        def param_dalmp(m, t, s):
            return data[s][t][0]
        def param_windgen(m, t, s):
            return data[s][t][2]
        def windforecast(m, t):
            return data[:,t,2].mean()

        # Param 
        m.rtlmp = pyo.Param( (m.times * m.scenarios), within= pyo.Reals, mutable=True, initialize=param_rtlmp)
        m.dalmp = pyo.Param( (m.times * m.scenarios), within= pyo.Reals, mutable=True, initialize=param_dalmp)
        m.windgen = pyo.Param( (m.times * m.scenarios), within= pyo.Reals, mutable=True, initialize=param_windgen)
        m.windforecast = pyo.Param( (m.times), within= pyo.Reals, mutable=True, initialize=windforecast)
        m.wind_capacity_mw = pyo.Param(within= pyo.Reals, initialize=wind_capacity_mw, mutable=True)
        m.bigM = pyo.Expression(rule= lambda m:m.wind_capacity_mw * 10)  # TODO: make Pyomo assign best bigM values 

        # Variables 
        m.quantity_offer = pyo.Var(m.times, within= pyo.Reals, initialize = 0)
        m.quantity_offer.setlb(-m.wind_capacity_mw) # TODO: when wind_capacity_mw changes, test whether m.quantity_offer upper and lower bound would also update 
        m.quantity_offer.setub(m.wind_capacity_mw)

        m.CONSTR_wind_farm_forecast_offer_ub = pyo.Constraint( (m.times * m.scenarios), rule=lambda m,t,s:
                                                     m.quantity_offer[t]  <= m.windforecast[t] )
        # m.CONSTR_wind_farm_forecast_offer_lb = pyo.Constraint( (m.times * m.scenarios), rule=lambda m,t,s:
        #                                              m.quantity_offer[t]  >= -m.windforecast[t])


        # Expressions for unit revenue and scenario revenue 
        def per_scenario_per_time_revenue(m, t, s):
            return (
                        (m.dalmp[t,s] - m.rtlmp[t,s]) * (m.quantity_offer[t]) +
                        m.windgen[t,s] * m.rtlmp[t,s]
            )

        m.unit_revenue = pyo.Expression((m.times * m.scenarios), rule = per_scenario_per_time_revenue)
        m.revenue = pyo.Expression((m.scenarios), rule =
                                            lambda m, s: sum(m.unit_revenue[t,s] for t in m.times)
        )

        # CVar 
        #        param
        m.beta = pyo.Param( within= pyo.Reals, initialize=0.5, mutable=True)
        m.alpha = pyo.Param( within= pyo.Reals, initialize=0.95, mutable=True)
        #        variables 
        m.eta = pyo.Var(m.scenarios, domain=pyo.NonNegativeReals)
        m.zeta = pyo.Var(domain=pyo.Reals)
        #        expressions
        m.cvar = pyo.Expression(rule= lambda m:  m.zeta - 1/(1-m.alpha) * sum( 1/len(m.scenarios) * m.eta[s] for s in m.scenarios
                ))
        #        constraints 
        m.CONSTR_cvar = pyo.Constraint(m.scenarios, rule=lambda m, s:
                                    -sum(m.unit_revenue[t,s] for t in m.times) + m.zeta - m.eta[s] <= 0
                                    )

        # Objective 
        m.OBJ_cvar = pyo.Objective(rule= lambda m:  (
            (1-m.beta) * sum( 1/len(m.scenarios) *  m.revenue[s] for s in m.scenarios) +
                m.beta * (m.zeta - 1/(1-m.alpha) * sum( 1/len(m.scenarios) * m.eta[s] for s in m.scenarios ))
                ),
            sense = pyo.maximize)

        return m 

    def build_dual_econ_bid_model(self) -> pyo.ConcreteModel:
        # disable self-schedule cvar constraints and objectives 
        m = self.build_self_schedule_model()
        m.CONSTR_cvar.deactivate()
        m.OBJ_cvar.deactivate()

        # Var 
        # delete offer and bid cleared component quantity at each time and scenarios  
        m.quantity_bid_scenario = pyo.Var( (m.times * m.scenarios), within= pyo.Reals, bounds = (0, m.wind_capacity_mw), initialize=0) # introduce separate variable for realized quantity offer
        m.quantity_offer_scenario = pyo.Var( (m.times * m.scenarios), within= pyo.Reals, bounds = (0, m.wind_capacity_mw), initialize=0) # introduce separate variable for realized quantity bid

        # m.quantity_offer = pyo.Var( (m.times), within= pyo.Reals, bounds = (0, m.wind_capacity_mw)) 
        m.quantity_offer.setlb(0) 
        m.quantity_bid = pyo.Var( (m.times), within= pyo.Reals, bounds = (0, m.wind_capacity_mw), initialize =0) # introduce separate variable for bid

        # m.price_offer = pyo.Param(m.times, within= pyo.Reals, initialize=0, mutable=True)
        m.price_offer = pyo.Var(m.times, within= pyo.Reals, initialize=0)
        m.sigma_offer = pyo.Var( (m.times * m.scenarios), within = pyo.Binary)
        m.price_bid = pyo.Var(m.times, within= pyo.Reals, initialize=0)
        m.sigma_bid = pyo.Var( (m.times * m.scenarios), within = pyo.Binary)


        # ======
        # Offer and Bid price oncstraints 
        # ======

        def offer_clear_rule_price_scenarios(m, t, s):
            return m.price_offer[t] <=  m.dalmp[t,s] - m.bigM * (1 - m.sigma_offer[t,s]) 
            # price_offer <= dalmp. when sigma_offer is 1, this constraint holds 
        

        def bid_clear_rule_price_scenarios(m, t, s):
            return m.price_bid[t] >=  m.dalmp[t,s] + m.bigM * (1 - m.sigma_bid[t,s]) 
            # price_bid >= dalmp. when sigma_bid is 1, this constraint holds 

        m.CONSTR_offer_clear_price_per_scenarios = pyo.Constraint( (m.times * m.scenarios), rule = offer_clear_rule_price_scenarios)
        m.CONSTR_bid_clear_price_per_scenarios = pyo.Constraint( (m.times * m.scenarios), rule = bid_clear_rule_price_scenarios)


        # half of the offers cleared 
        # m.CONSTR_bullshit = pyo.Constraint(rule = lambda m: sum(m.sigma_offer[t,s] + m.sigma_bid[t,s] for t in m.times for s in m.scenarios) >= 
        #                                    sum(2 for t in m.times for s in m.scenarios) * 0.1 )

        # m.CONSTR_sigma = pyo.Constraint((m.times * m.scenarios), rule = lambda m, t, s: m.sigma_offer[t,s] + m.sigma_bid[t,s] == 1 )
        m.CONSTR_sigma = pyo.Constraint((m.times * m.scenarios), rule = lambda m, t, s: m.sigma_offer[t,s] + m.sigma_bid[t,s] <= 1 ) # can only award either bid or offer. 

        # ======
        # Offer quantity oncstraints 
        # ======
        # when sigma offer is 1, then m.quantity_offer[t] >= m.quantity_offer_scenario[t,s] >= m.quantity_offer[t]
        # when sigma offer is 0,      M >= m.quantity_offer_scenario[t,s] >= - M
        # m.quantity_offer[t] + m.bigM * ( 1- m.sigma_offer[t,s])  >= m.quantity_offer_scenario[t,s] >= m.quantity_offer[t] - m.bigM * ( 1- m.sigma_offer[t,s])        
        
        # when sigma offer is 0, then     0    >= m.quantity_offer_scenario >= 0 
        # when sigma offer is 1, then     M    >= m.quantity_offer_scenario >= -M 
        #  + m.bigM *  m.sigma_offer[t,s]  >= m.quantity_offer_scenario[t,s] >=  m.bigM * m.sigma_offer[t,s])

        # bigM to dictate whether it clears or not per time per scenarios  
        def offer_clear_rule_1(m, t, s):
            return      m.quantity_offer_scenario[t,s] >= m.quantity_offer[t] - m.bigM * ( 1- m.sigma_offer[t,s])
        def offer_clear_rule_2(m, t, s):
            return  m.quantity_offer[t] + m.bigM * ( 1- m.sigma_offer[t,s]) >= m.quantity_offer_scenario[t,s]
        def offer_clear_rule_3(m, t, s):
            return m.quantity_offer_scenario[t,s] >=  0 -  m.bigM * m.sigma_offer[t,s]
        def offer_clear_rule_4(m, t, s):
            return m.bigM *  m.sigma_offer[t,s]  >= m.quantity_offer_scenario[t,s]
    
        m.CONSTR_offer_clear_rule_1 = pyo.Constraint( (m.times * m.scenarios), rule = offer_clear_rule_1)
        m.CONSTR_offer_clear_rule_2 = pyo.Constraint( (m.times * m.scenarios), rule = offer_clear_rule_2)
        m.CONSTR_offer_clear_rule_3 = pyo.Constraint( (m.times * m.scenarios), rule = offer_clear_rule_3)
        m.CONSTR_offer_clear_rule_4 = pyo.Constraint( (m.times * m.scenarios), rule = offer_clear_rule_4)

        # ======
        # Bid quantity oncstraints 
        # ======
        # when sigma bid is 1, then m.quantity_bid[t] >= m.quantity_bid_scenario[t,s] >= m.quantity_bid[t]
        # when sigma bid is 0,      M >= m.quantity_bid_scenario[t,s] >= - < 
        # m.quantity_bid[t] + m.bigM * ( 1- m.sigma_bid[t,s])  >= m.quantity_bid_scenario[t,s] >= m.quantity_bid[t] - m.bigM * ( 1- m.sigma_bid[t,s])        

        # when sigma bid is 0, then     0    >= m.quantity_bid_scenario >= 0 
        # when sigma bid is 1, then     M    >= m.quantity_bid_scenario >= -M 
        #  + m.bigM *  m.sigma_bid[t,s]  >= m.quantity_bid_scenario[t,s] >=  m.bigM * m.sigma_bid[t,s])

        # bigM to dictate whether it clears or not per time per scenarios  
        def bid_clear_rule_1(m, t, s):
            return      m.quantity_bid_scenario[t,s]   >= m.quantity_bid[t] - m.bigM * ( 1- m.sigma_bid[t,s])
        def bid_clear_rule_2(m, t, s):
            return  m.quantity_bid[t] + m.bigM * ( 1- m.sigma_bid[t,s])  >= m.quantity_bid_scenario[t,s]
        def bid_clear_rule_3(m, t, s):
            return m.quantity_bid_scenario[t,s] >=  0 - m.bigM * m.sigma_bid[t,s]
        def bid_clear_rule_4(m, t, s):
            return m.bigM *  m.sigma_bid[t,s]  >= m.quantity_bid_scenario[t,s]

        m.CONSTR_bid_clear_rule_1 = pyo.Constraint( (m.times * m.scenarios), rule = bid_clear_rule_1)
        m.CONSTR_bid_clear_rule_2 = pyo.Constraint( (m.times * m.scenarios), rule = bid_clear_rule_2)
        m.CONSTR_bid_clear_rule_3 = pyo.Constraint( (m.times * m.scenarios), rule = bid_clear_rule_3)
        m.CONSTR_bid_clear_rule_4 = pyo.Constraint( (m.times * m.scenarios), rule = bid_clear_rule_4)


        m.CONSTR_offer_clear_quantity = pyo.Constraint( (m.times * m.scenarios), rule = lambda m, t, s: m.quantity_offer_scenario[t,s] <=  m.windforecast[t] )
        # m.CONSTR_bid_clear_quantity = pyo.Constraint( (m.times * m.scenarios), rule = lambda m, t, s: m.quantity_bid_scenario[t,s] <=  m.windforecast[t])

        # p_opt, _ = find_p_argmax(self.data, False)  # 
        # m.price_offer.store_values(p_opt)

        # Expressions for unit revenue and scenario revenue 
        def unit_revenue_econ(m, t, s):
            return (
                        (m.dalmp[t,s] - m.rtlmp[t,s]) * m.quantity_offer_scenario[t,s] +
                        (m.rtlmp[t,s] - m.dalmp[t,s]) * m.quantity_bid_scenario[t,s] +
                        m.windgen[t,s] * m.rtlmp[t,s]
            )
        m.unit_revenue_econ = pyo.Expression((m.times * m.scenarios), rule = unit_revenue_econ)
        m.revenue_econ = pyo.Expression((m.scenarios), rule =
                                            lambda m, s: sum(m.unit_revenue_econ[t,s] for t in m.times)
        )

        # CVar constarint for economic bid strategy  
        m.CONSTR_economic_bid_cvar = pyo.Constraint(m.scenarios, rule=lambda m, s:
                                    -sum(m.unit_revenue_econ[t,s] for t in m.times) + m.zeta - m.eta[s] <= 0
                                    )

        # Objective for economic bid strategy 

        m.OBJ_economic_bid_cvar = pyo.Objective(rule= lambda m:  (
            (1-m.beta) * sum( 1/len(m.scenarios) *  m.revenue_econ[s] for s in m.scenarios) +
                m.beta * (m.zeta - 1/(1-m.alpha) * sum( 1/len(m.scenarios) * m.eta[s] for s in m.scenarios ))
                ),
            sense = pyo.maximize)

        return m 


    def build_econ_bid_model(self, m:pyo.ConcreteModel) -> pyo.ConcreteModel:

        # disable self-schedule cvar constraints and objectives 
        m.CONSTR_cvar.deactivate()
        m.OBJ_cvar.deactivate()

        # Var 
        m.quantity_bid = pyo.Var(m.times, within= pyo.Reals, bounds = (0, m.wind_capacity_mw), initialize=0) # introduce separate variable for bid
        m.quantity_offer.setlb(0)

        # Param
        m.price = pyo.Param(m.times, within= pyo.Reals, initialize=0, mutable=True)
        m.price_indicator = pyo.Param( (m.times * m.scenarios), mutable=True)

        # bigM to model disjunction on whether to put in an offer or a bid  
        def offer_clear_rule_quantity(m, t):
            return m.quantity_offer[t] <= m.bigM * m.binary_offer_clear[t]
        def bid_clear_rule_quantity(m, t):
            return m.quantity_bid[t] <= m.bigM * (1 - m.binary_offer_clear[t])
        m.binary_offer_clear = pyo.Var((m.times), within = pyo.Binary)
        m.CONSTR_offer_clear_quantity = pyo.Constraint( (m.times), rule = offer_clear_rule_quantity)
        m.CONSTR_bid_clear_quantity = pyo.Constraint( (m.times), rule = bid_clear_rule_quantity)


        # m.CONSTR_wind_farm_forecast_bid = pyo.Constraint( (m.times * m.scenarios), rule=lambda m,t,s:
        #                                              m.quantity_bid[t]  <= m.windforecast[t])





        # Expressions for unit revenue and scenario revenue 
        def unit_revenue_econ(m, t, s):
            return (
                        (m.dalmp[t,s] - m.rtlmp[t,s]) * m.price_indicator[t,s] * m.quantity_offer[t] +
                        (m.rtlmp[t,s] - m.dalmp[t,s]) * (1 - m.price_indicator[t,s]) * m.quantity_bid[t] +
                        m.windgen[t,s] * m.rtlmp[t,s]
            )
        m.unit_revenue_econ = pyo.Expression((m.times * m.scenarios), rule = unit_revenue_econ)
        m.revenue_econ = pyo.Expression((m.scenarios), rule =
                                            lambda m, s: sum(m.unit_revenue_econ[t,s] for t in m.times)
        )

        # CVar constarint for economic bid strategy  
        m.CONSTR_economic_bid_cvar = pyo.Constraint(m.scenarios, rule=lambda m, s:
                                    -sum(m.unit_revenue_econ[t,s] for t in m.times) + m.zeta - m.eta[s] <= 0
                                    )

        # Objective for economic bid strategy 

        m.OBJ_economic_bid_cvar = pyo.Objective(rule= lambda m:  (
            (1-m.beta) * sum( 1/len(m.scenarios) *  m.revenue_econ[s] for s in m.scenarios) +
                m.beta * (m.zeta - 1/(1-m.alpha) * sum( 1/len(m.scenarios) * m.eta[s] for s in m.scenarios ))
                ),
            sense = pyo.maximize)

        return m 


    def update_model(self, new_strat: str):
        econ_bid_stra = [ECONOMIC_BID_P_ARGMAX, ECONOMIC_BID_P_MEAN]
        old_strat, new_strat = self.strategy, new_strat
        self.set_strategy(new_strat)
        if old_strat in econ_bid_stra and new_strat in econ_bid_stra:  
            # both economic bid models are the same, except for optimal pricing strategy difference 
            self.update_price() 
        elif old_strat == SELF_SCHEDULE and \
            new_strat in econ_bid_stra:
            self.expand_model()
        else:                            
            self.build_model()

    def update_data(self, new_data):
        self.data = new_data
        self.load_dalmp_rtlmp_windgen(new_data)
        self.update_price()

    def update_price(self):
        if self.strategy == ECONOMIC_BID_P_MEAN:
            p_opt, _ = find_p_mean_rtlmp(self.data)
        elif self.strategy == ECONOMIC_BID_P_ARGMAX:
            p_opt, _ = find_p_argmax(self.data) 
        # load new prices 
        self.load_price(p_opt)

    def load_price(self, new_price_dict: Dict[str, float]):
        m = self.model 
        m.price.store_values(new_price_dict)
        for t in m.times:
            for s in m.scenarios:
                m.price_indicator[t,s].value = 1 if m.dalmp[t, s].value >= m.price[t].value else 0

    def load_cvar_alpha_beta(self, cvar: Dict[str, float]):
        if not (0 <= cvar['alpha'] <= 1):
            raise ValueError(f"Alpha must be between 0 and 1, but got {cvar['alpha']}")
        if not (0 <= cvar['beta'] <= 1):
            raise ValueError(f"Beta must be between 0 and 1, but got {cvar['beta']}")

        self.model.alpha.value = cvar['alpha']
        self.model.beta.value = cvar['beta']

    def load_dalmp_rtlmp_windgen(self, data: np.array):

        expected_shape = (len(self.model.scenarios), len(self.model.times), 3)
        if data.shape != expected_shape:
            raise ValueError(f"Expected data shape {expected_shape}, but got {data.shape}")

        rtlmp_values = {
            (i, j): data[s][t][1]
            for t, i in enumerate(self.model.times)
            for s, j in enumerate(self.model.scenarios)
        }
        self.model.rtlmp.store_values(rtlmp_values)


        dalmp_values = {
            (i, j): data[s][t][0]
            for t, i in enumerate(self.model.times)
            for s, j in enumerate(self.model.scenarios)
        }
        self.model.dalmp.store_values(dalmp_values)


        windgen_values = {
            (i, j): data[s][t][2]
            for t, i in enumerate(self.model.times)
            for s, j in enumerate(self.model.scenarios)
        }

        self.model.windgen.store_values(windgen_values)

