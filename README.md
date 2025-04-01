
# Offer-Bid-Opt: MILP-based Model for Bidding Strategies with CVaR Risk Management

This project provides simple MILP-based models to calculate optimal offering/bidding strategies (price x quantity) for Variable Renewable Energy (VRE) generation, with CVaR (Conditional Value-at-Risk) risk management.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Running Tests](#running-tests)
4. [License](#license)

## Installation

### Prerequisites

- Python 3.7+ 
- uv  

## 1. Installation


 **Install dependencies**:

get uv 
   ```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

install dependency 
   ```bash
uv sync
   ```

**Optional**: If you need specific solver installations (e.g., IPOPT, GLPK), follow the instructions in their respective documentation.


## 2. Usage

After installation, you can start using the modules as follows:

### Example Code

```python
from offer_bid_opt.bidding_optimizer import *
from offer_bid_opt.sample_data_generator import *
from offer_bid_opt.constants import *

bidding_instance = BiddingOptimizer()

data, _ = generate_sample_data(300, 24, 150)
bidding_instance.set_data(data)  # dalmp, rtlmp, windgen
bidding_instance.set_wind_capacity_mw(150)

# Solver options (Example)
solver_option = ['ipopt', 'bin/ipopt', False]  # Or ['glpk', None, False]

# Set strategy
# SELF_SCHEDULE = 0
# ECONOMIC_BID_P_MEAN = 1
# ECONOMIC_BID_P_ARGMAX = 2
# ECONOMIC_BID_DUAL = 3
bidding_instance.set_strategy(new_strat=SELF_SCHEDULE)

# Solve with CVaR parameters
bidding_instance.optimize(visualize_option=True, cvar={'alpha':0.8, 'beta':0}, solver_option=solver_option)
bidding_instance.optimize(visualize_option=True, cvar={'alpha':0.95, 'beta':0.01}, solver_option=solver_option)
bidding_instance.optimize(visualize_option=True, cvar={'alpha':0.95, 'beta':0.8}, solver_option=solver_option)
bidding_instance.optimize(visualize_option=True, cvar={'alpha':0.95, 'beta':1}, solver_option=solver_option)
```

## 3. Running Tests

To run the tests, you can use the following command:

```bash
uv run pytest
```

This will execute all the test cases and display the results in the terminal.

## 4. License
This project is licensed under the MIT License.

## Contributing
Feel free to contribute by submitting issues or pull requests.

## Directory Structure

```plaintext
offer_bid_opt/
│- test/                     # Unit tests for the bidding optimizer and utilities
│  |- test_bidding_optimizer.py
│  |- test_utility.py
│  |- test_price.py
│
│- script/                   # Scripts for running different strategies
│  |- run_strategies.py
│  |- run_strategies_limit_quantity.py
│  |- run_optimal_prices.py
│
│- offer_bid_opt/             # Core package for optimization and utilities
│  |- __init__.py
│  |- utility.py
│  |- constants.py
│  |- postprocess.py
│  |- visualize.py
│  |- sample_data_generator.py
│  |- bidding_model.py
│  |- bidding_optimizer.py
│  |- price.py
│
│- data/                     # Sample datasets for testing and analysis
│  |- energy_market_samples_large.npz
│  |- energy_market_samples.npz
```

### `offer_bid_opt`
- **bidding_optimizer.py**: Interface to the underlying models and optimizations.
- **bidding_model.py**: Defines mathematical models for bidding.
- **price.py**: Handles optimal price calculations.
- **utility.py**: Utility functions for data processing.
- **postprocess.py**: Processes optimization results.
- **visualize.py**: Visualization tools for offer/bid analysis.
- **constants.py**: Stores project-wide strategy constants.
- **sample_data_generator.py**: Generates sample market data.

### `script`
- **run_strategies.py**: Runs bidding strategy simulations.
- **run_strategies_limit_quantity.py**: Executes strategy with quantity constraints.
- **run_optimal_prices.py**: Computes optimal prices based on market data.

### `test`
- Contains unit tests.

### `data`
- Contains sample market data in `.npz` format.

