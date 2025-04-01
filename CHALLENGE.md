# Senior Optimization Data Science Role - Take-Home Challenge

We've designed this challenge to give you an opportunity to showcase your technical skills and optimization modeling in a realistic scenario that aligns with our work. Here are the details:

## Challenge Overview
You'll develop an optimization model for trading day-ahead (DA) versus real-time (RT) markets for a renewable generation asset (Wind). The output of the model is a single bid and offer price and quantity pair (PQ pair) for each hour. The model should focus on the relationship between DA/RT price spreads across the day and identify opportunities for arbitrage and risk management. We've provided code to generate synthetic data including Locational Marginal Prices (LMP) and wind generation values that will form the foundation of your optimization strategy.

## Market Clearing and Settlement

For a given hour, market settlement follows these rules:

### Revenue Calculations
- **Settled Offer Revenue** = Cleared DA Offer Volume × (DA Price - RT Price)
- **Settled Bid Revenue** = Cleared DA Bid Volume × (-DA Price + RT Price)
- **Wind Revenue** = Wind Production Power × RT Price

### Clearing Conditions
- An **offer** clears if the DA offer price is **greater than or equal to** the DA cleared price
- A **bid** clears if the DA bid price is **less than or equal to** the DA cleared price

### Total Revenue
Total Revenue = Settled Offer Revenue + Settled Bid Revenue + Wind Revenue

## Requirements

### Technical Implementation
* Develop a well-structured Python package demonstrating:
   * OOP principles through a robust class implementation
   * Comprehensive test coverage
   * Production-quality documentation

### Optimization Model
* Based on your data analysis:
   * Clearly define your optimization problem formulation
   * Demonstrate how your approach manages risk/reward tradeoffs
   * Assume access to well-calibrated, probabilistic price and generation forecasts at the day-ahead stage
   * Specify technologies/libraries you'd use and justify your choices
   * Outline implementation risks and proposed mitigation strategies

## Deliverables
* Your Python implementation (feel free to share in your preferred format - Jupyter notebook, Python files, etc.)
* A brief technical report outlining your approach, findings, and potential extensions
* A development plan for the optimization model, including problem formulation and risk analysis

## Time Frame
* We recommend spending around 2-3 hours on this
* Please submit by Monday, 24th March

## Environment Setup
We've provided a `pyproject.toml` file to help you set up your development environment with all necessary dependencies:

1. Extract the provided zip file
2. Install the package and dependencies using UV (a faster Python package installer):
   ```bash
   # Install dependencies
   uv sync
   
   # For development dependencies
   uv sync --dev
   ```
   
   If you don't have UV installed, you can install it with:
   ```bash
   # On macOS/Linux
   curl -sSf https://install.determinate.systems/uv | bash
   
   # On Windows PowerShell
   irm https://install.determinate.systems/uv | iex
   ```

3. Start JupyterLab to begin your work:
   ```bash
   uv run jupyter-lab
   ```
   This will open a JupyterLab environment in your browser where you can create notebooks for developing your optimization model.

## Data Generation
We've provided a Python script (`sample_data_generator.py`) that creates synthetic energy market data for a single node. This script generates:

* Day-Ahead Locational Marginal Prices (DALMP)
* Real-Time Locational Marginal Prices (RTLMP)
* Wind Power generation values

The output is a numpy array with shape (100, 24, 3) representing:
* 100 ensemble samples/scenarios (all for the same day)
* 24 hours in the day
* 3 target variables (DALMP, RTLMP, Wind generation)

**Important concept:** All samples represent different possible scenarios for the SAME day, not different days. This simulates the uncertainty in forecasting for a single upcoming day.

To generate the sample data:
1. Run the script using UV: `uv run python sample_data_generator.py`
2. The script will create a `data` directory (if it doesn't exist) and save a file named `energy_market_samples.npz`
3. Use this numpy file as the data source for your optimization model

## Using the Sample Data in Your Work
You can import the sample data generator functions in your scripts or notebooks:

```python
# Import the functions from the script
from sample_data_generator import generate_sample_data, load_sample_data

# Generate new data
data, target_names = generate_sample_data()

# Or load existing data
data, target_names = load_sample_data()
```

## Folder Structure
After extracting the zip file, you should have the following structure:

```
Senior DS - Optimization/
├── data/                            # Created when sample data is generated
│   └── energy_market_samples.npz    # Generated sample data
├── CHALLENGE.md                     # This document
├── pyproject.toml                   # Project configuration and dependencies
└── sample_data_generator.py         # Script to generate sample data
```

We understand that you're likely balancing multiple commitments, so please let us know if you need any adjustments to the timeline. If you have any questions about the challenge or need clarification, don't hesitate to reach out.