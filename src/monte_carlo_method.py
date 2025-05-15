"""
Monte Carlo Simulation Method for VaR and ES Calculation.

This module implements the logic to calculate Value-at-Risk (VaR) and
Expected Shortfall (ES) using Monte Carlo simulation.
"""
import numpy as np
# from .utils import convert_annual_to_daily # If needed and not passed differently

def calculate_monte_carlo_var_es(portfolio_config: dict,
                                 # Pre-calculated daily figures can be passed for efficiency
                                 daily_returns: np.ndarray,
                                 daily_vols: np.ndarray,
                                 daily_covariance_matrix: np.ndarray): # or L_matrix
    """
    Calculates VaR and ES using the Monte Carlo simulation method.

    Args:
        portfolio_config (dict): A dictionary containing parameters:
            - 'portfolio_value': float
            - 'weights': np.ndarray
            - 'confidence_level': float
            - 'time_horizon_days': int
            - 'num_simulations': int
        daily_returns (np.ndarray): Expected daily returns for each asset.
        daily_vols (np.ndarray): Expected daily volatilities for each asset.
                                 (Used here if L_matrix is not pre-computed from daily_cov)
        daily_covariance_matrix (np.ndarray): Daily covariance matrix of asset returns.
                                              Used to derive Cholesky factor L.

    Returns:
        tuple: (var_value, es_value, var_return, es_return, all_sim_returns)
               - var_value (float): Value-at-Risk as a positive loss value.
               - es_value (float): Expected Shortfall as a positive loss value.
               - var_return (float): VaR as a negative return.
               - es_return (float): ES as a negative return.
               - all_sim_returns (np.ndarray): Array of all simulated portfolio returns
                                               over the horizon for plotting/analysis.
    Assumptions:
        - Asset returns can be modeled by a multivariate normal distribution
          (or other specified distribution if generation logic is changed).
        - Daily returns are IID for multi-day simulation (compounding).
    """
    # Unpack parameters
    portfolio_value = portfolio_config['portfolio_value']
    weights = portfolio_config['weights']
    confidence_level = portfolio_config['confidence_level']
    time_horizon = portfolio_config['time_horizon_days']
    num_simulations = portfolio_config['num_simulations']
    num_assets = len(weights)

    alpha = 1 - confidence_level

    # 1. Generate Correlated Daily Asset Returns
    # Cholesky Decomposition of the daily covariance matrix: L L^T = Sigma_daily
    try:
        L_matrix = np.linalg.cholesky(daily_covariance_matrix)
    except np.linalg.LinAlgError:
        # Fallback or error handling if matrix is not positive definite
        # For simplicity, we'll raise an error. In practice, might try to find nearest PD matrix.
        raise ValueError("Daily covariance matrix is not positive definite. Cholesky decomposition failed.")

    # Array to store simulated end-of-horizon portfolio values or returns
    sim_portfolio_end_values = np.zeros(num_simulations)
    # Alternative: store horizon returns directly
    sim_horizon_portfolio_returns = np.zeros(num_simulations)


    for i in range(num_simulations):
        current_portfolio_value_sim = portfolio_value # Start with initial value for this path

        for _ in range(time_horizon): # Simulate each day in the horizon
            # Generate 'num_assets' independent standard normal random numbers (Z_day)
            standard_random_numbers = np.random.normal(0, 1, size=num_assets)
            
            # Transform to correlated daily asset returns for the current day:
            # Delta_R_assets_day = expected_daily_returns + L * Z_day
            sim_daily_asset_returns = daily_returns + L_matrix @ standard_random_numbers
            
            # Calculate simulated portfolio return for the current day:
            # Delta_R_portfolio_day = w^T * Delta_R_assets_day
            sim_daily_portfolio_return = np.dot(weights, sim_daily_asset_returns)
            
            # Update the portfolio value for this simulation path for this day
            current_portfolio_value_sim *= (1 + sim_daily_portfolio_return)
        
        # Store the final portfolio value for this simulation path
        sim_portfolio_end_values[i] = current_portfolio_value_sim
        # Or calculate and store the total return over the horizon for this path
        sim_horizon_portfolio_returns[i] = (current_portfolio_value_sim / portfolio_value) - 1


    # 2. Calculate Simulated Portfolio P&L and Returns (if using end_values)
    # sim_portfolio_pnl = sim_portfolio_end_values - portfolio_value
    # sim_portfolio_returns = sim_portfolio_pnl / portfolio_value
    # We are using sim_horizon_portfolio_returns directly

    # 3. Calculate VaR from Simulated Portfolio Returns
    # Sort the sim_horizon_portfolio_returns in ascending order.
    sorted_sim_returns = np.sort(sim_horizon_portfolio_returns)
    
    # Find the return at the alpha percentile (this is the VaR return)
    var_index = int(alpha * num_simulations) # Index for the (alpha*100)th percentile
    
    # Ensure index is within bounds, especially if alpha or num_simulations is very small
    var_index = max(0, min(var_index, num_simulations - 1))

    var_mc_return = sorted_sim_returns[var_index]
    
    # VaR (as a positive loss value)
    var_mc_value = -var_mc_return * portfolio_value
    var_mc_value = max(0, var_mc_value) # Ensure non-negative loss


    # 4. Calculate ES from Simulated Portfolio Returns
    # ES is the average of returns that are worse than or equal to the VaR return.
    # These are the returns from index 0 up to var_index (inclusive).
    if var_index == 0 and sorted_sim_returns[0] >= 0 : # No losses in the tail
         es_mc_return = sorted_sim_returns[0] # Smallest return, could be positive
    elif var_index < num_simulations -1 :
        es_mc_return = np.mean(sorted_sim_returns[:var_index + 1])
    else: # all simulations resulted in losses worse than var_index (highly unlikely for typical alpha)
        es_mc_return = np.mean(sorted_sim_returns)


    # ES (as a positive loss value)
    es_mc_value = -es_mc_return * portfolio_value
    es_mc_value = max(0, es_mc_value) # Ensure non-negative loss

    return var_mc_value, es_mc_value, var_mc_return, es_mc_return, sorted_sim_returns


if __name__ == '__main__':
    print("Testing Monte Carlo VaR/ES Calculation Module...")
    # For direct testing, we need a sample config and pre-calculated daily figures
    # This setup is a bit more involved than parametric due to dependencies.

    # Simplified daily figures for testing purposes
    test_daily_returns = np.array([0.10/252, 0.05/252])
    test_daily_vols = np.array([0.20/np.sqrt(252), 0.10/np.sqrt(252)])
    test_corr_matrix = np.array([[1.0, 0.3], [0.3, 1.0]])
    
    D_test = np.diag(test_daily_vols)
    test_daily_cov_matrix = D_test @ test_corr_matrix @ D_test

    sample_mc_config_for_test = {
        "name": "MC Test Portfolio",
        "portfolio_value": 1_000_000,
        "weights": np.array([0.6, 0.4]),
        "confidence_level": 0.99,
        "time_horizon_days": 10,
        "num_simulations": 50000 # Smaller for quick test
        # Other params like annual returns/vols not directly used if daily figures are passed
    }
    
    # Set a seed for reproducibility during testing
    np.random.seed(42)

    mc_var_val, mc_es_val, mc_var_ret, mc_es_ret, _ = calculate_monte_carlo_var_es(
        sample_mc_config_for_test,
        test_daily_returns,
        test_daily_vols, # Not directly used if daily_cov_matrix is correct
        test_daily_cov_matrix
    )

    print(f"\n--- Monte Carlo Method Test Results (Seed=42) ---")
    print(f"Portfolio Value: ${sample_mc_config_for_test['portfolio_value']:,.2f}")
    print(f"Confidence Level: {sample_mc_config_for_test['confidence_level']*100:.1f}%")
    print(f"Time Horizon: {sample_mc_config_for_test['time_horizon_days']} days")
    print(f"Simulations: {sample_mc_config_for_test['num_simulations']}")
    print("-" * 30)
    print(f"VaR Return: {mc_var_ret:,.4%}")
    print(f"VaR Value: ${mc_var_val:,.2f}")
    print(f"ES Return: {mc_es_ret:,.4%}")
    print(f"ES Value: ${mc_es_val:,.2f}")
    print("-" * 30)
