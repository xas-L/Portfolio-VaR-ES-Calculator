"""
Main Orchestration Script for Portfolio Risk Calculator.

This script loads portfolio configurations, performs VaR and ES calculations
using different methods (Parametric, Monte Carlo), and displays the results.
"""
import numpy as np
import matplotlib.pyplot as plt
from src.utils import display_results, convert_annual_to_daily
from src.parametric_method import calculate_parametric_var_es
from src.monte_carlo_method import calculate_monte_carlo_var_es

# --- Embedded Portfolio Configuration ---
# Moved DEFAULT_PORTFOLIO from config.portfolio_config into this file
# to resolve ModuleNotFoundError for 'config' in certain execution environments.
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
# --- End of Embedded Portfolio Configuration ---


def run_risk_calculations(portfolio_config_name: str = "DEFAULT_PORTFOLIO"):
    """
    Runs the risk calculations for a specified portfolio configuration.

    Args:
        portfolio_config_name (str): The name of the portfolio configuration
                                     to use. Currently, only "DEFAULT_PORTFOLIO"
                                     is directly embedded.
    """
    print(f"--- Starting Risk Calculations for: {portfolio_config_name} ---")

    # --- 1. Load Portfolio Configuration ---
    if portfolio_config_name == "DEFAULT_PORTFOLIO":
        current_config = DEFAULT_PORTFOLIO
    else:
        # Placeholder for loading other configs if they were defined elsewhere
        # For now, this example only uses the embedded DEFAULT_PORTFOLIO
        print(f"Warning: Portfolio '{portfolio_config_name}' not found. Using DEFAULT_PORTFOLIO.")
        current_config = DEFAULT_PORTFOLIO

    # --- 2. Preparations for Monte Carlo (Daily Figures) ---
    # These are also used by the parametric method internally, but MC needs them explicitly.
    # It's good to calculate them once.
    daily_returns_mc = convert_annual_to_daily(
        current_config['expected_annual_returns'],
        current_config['trading_days_per_year'],
        is_volatility=False
    )
    daily_vols_mc = convert_annual_to_daily(
        current_config['annual_volatilities'],
        current_config['trading_days_per_year'],
        is_volatility=True
    )
    D_daily_mc = np.diag(daily_vols_mc)
    daily_covariance_matrix_mc = D_daily_mc @ current_config['correlation_matrix'] @ D_daily_mc


    # --- 3. Parametric VaR & ES Calculation ---
    try:
        param_var_val, param_es_val, param_var_ret, param_es_ret = \
            calculate_parametric_var_es(current_config)
        
        display_results(
            method_name="Parametric (Variance-Covariance)",
            var_value=param_var_val,
            es_value=param_es_val,
            var_return=param_var_ret,
            es_return=param_es_ret,
            portfolio_config=current_config
        )
    except Exception as e:
        print(f"Error during Parametric calculation: {e}")


    # --- 4. Monte Carlo VaR & ES Calculation ---
    print(f"\nStarting Monte Carlo simulation with {current_config['num_simulations']} paths...")
    print("(This may take a moment depending on the number of simulations and horizon)")
    
    # Set a seed for reproducibility of Monte Carlo results if desired for runs
    # np.random.seed(42) # Uncomment if you want consistent MC results across runs

    try:
        mc_var_val, mc_es_val, mc_var_ret, mc_es_ret, all_sim_returns = \
            calculate_monte_carlo_var_es(
                portfolio_config=current_config,
                daily_returns=daily_returns_mc,
                daily_vols=daily_vols_mc, # Not strictly needed if daily_cov_matrix is passed
                daily_covariance_matrix=daily_covariance_matrix_mc
            )

        display_results(
            method_name="Monte Carlo Simulation",
            var_value=mc_var_val,
            es_value=mc_es_val,
            var_return=mc_var_ret,
            es_return=mc_es_ret,
            portfolio_config=current_config
        )

        # --- 5. Plot Histogram of Monte Carlo Returns ---
        # Plotting might not work in all execution environments (e.g., sandboxes without GUI)
        if all_sim_returns is not None:
            try:
                plt.figure(figsize=(10, 6))
                plt.hist(all_sim_returns, bins=100, alpha=0.75, color='skyblue', edgecolor='black',
                         label=f"Simulated Portfolio Returns ({current_config['time_horizon_days']}-day horizon)")
                
                plt.axvline(mc_var_ret, color='red', linestyle='dashed', linewidth=2,
                            label=f'VaR Return ({current_config["confidence_level"]*100:.1f}%): {mc_var_ret:,.4%}')
                plt.axvline(mc_es_ret, color='black', linestyle='dashed', linewidth=2,
                            label=f'ES Return ({current_config["confidence_level"]*100:.1f}%): {mc_es_ret:,.4%}')
                
                plt.title(f"Monte Carlo Simulation: Distribution of Portfolio Returns\n"
                          f"{current_config['name']} - {current_config['num_simulations']} Simulations")
                plt.xlabel(f"{current_config['time_horizon_days']}-Day Portfolio Return")
                plt.ylabel("Frequency")
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                # NB - You might want to save the plot to a file instead of showing it directly
                # plt.savefig("monte_carlo_returns_histogram.png")
                # plt.show() # This will block execution until the plot window is closed.
                print("\nPlot generated (plt.show() is commented out for non-interactive environments).")
                # In a script, plt.show() would typically be used. For sandboxes, saving might be better.
            except Exception as plot_e:
                print(f"Note: Plotting failed or was skipped. Error: {plot_e}")


    except Exception as e:
        print(f"Error during Monte Carlo calculation: {e}")

    print("\n--- Risk Calculations Finished ---")


if __name__ == "__main__":
 
    run_risk_calculations(portfolio_config_name="DEFAULT_PORTFOLIO")
