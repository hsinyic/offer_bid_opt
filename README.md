Simple MILP based model to calculate optimal offering/bidding strategies (price x quantity) for VRE
with CVaR risk management. 

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

## Contributing
Feel free to contribute by submitting issues or pull requests.

## License
This project is licensed under the MIT License.

```
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

