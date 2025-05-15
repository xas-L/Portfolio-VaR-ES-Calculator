# config/portfolio_config.py
"""
Portfolio Configuration File

This file defines the parameters for 1+ portfolios to be used
in the VaR and ES calculations. Multiple portfolio configurations can be defines here.
"""
import numpy as np

# --- Portfolio & Market Assumptions for a Sample Portfolio ---
# Multiple portfolio configurations can be defines as dictionaries
# or dataclasses. For simplicity, we'll start with one.

DEFAULT_PORTFOLIO = {
    "name": "Default Diversified Portfolio",
    "portfolio_value": 1_000_000,  # Initial portfolio value in currency units
    "asset_names": ['Equity_US', 'Bond_EU', 'Commodity_Gold'],

    # Asset weights in the portfolio (must sum to 1)
    "weights": np.array([0.5, 0.3, 0.2]),

    # Expected annualized returns for each asset
    "expected_annual_returns": np.array([0.08, 0.03, 0.05]), # e.g., 8%, 3%, 5%

    # Expected annualized volatilities (standard deviation) for each asset
    "annual_volatilities": np.array([0.15, 0.05, 0.18]), # e.g., 15%, 5%, 18%

    # Correlation matrix for asset returns
    "correlation_matrix": np.array([
        [1.0, 0.2, 0.1],  # Correlations of Equity_US with US, EU, Gold
        [0.2, 1.0, 0.05], # Correlations of Bond_EU with US, EU, Gold
        [0.1, 0.05, 1.0]  # Correlations of Commodity_Gold with US, EU, Gold
    ]),

    # --- Risk Parameters ---
    "confidence_level": 0.99,  # e.g., 99%
    "time_horizon_days": 10,     # VaR/ES time horizon in days

    # --- Monte Carlo Specific Parameters ---
    "num_simulations": 10000,   # Number of simulation paths for Monte Carlo

    # --- General Assumptions ---
    "trading_days_per_year": 252 # Used to convert annual figures to daily
}

# You could add more portfolios here, e.g.:
# AGGRESSIVE_PORTFOLIO = { ... }
# CONSERVATIVE_PORTFOLIO = { ... }

if __name__ == '__main__':
    # This part is just for testing the configuration file itself
    # and won't be run when imported by other modules.
    print("Portfolio Configuration Loaded:")
    for key, value in DEFAULT_PORTFOLIO.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}:")
            print(value)
        else:
            print(f"  {key}: {value}")

    # Basic validation example (can be expanded)
    if not np.isclose(np.sum(DEFAULT_PORTFOLIO["weights"]), 1.0):
        raise ValueError("Portfolio weights do not sum to 1.0")
    
    num_assets = len(DEFAULT_PORTFOLIO["asset_names"])
    if DEFAULT_PORTFOLIO["correlation_matrix"].shape != (num_assets, num_assets):
        raise ValueError(f"Correlation matrix shape is incorrect. Expected ({num_assets}, {num_assets})")

    print("\nBasic configuration validation passed.")
