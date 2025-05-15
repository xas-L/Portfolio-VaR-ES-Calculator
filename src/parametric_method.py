"""
Parametric (Variance-Covariance) Method for VaR and ES Calculation.

This module implements the logic to calculate Value-at-Risk (VaR) and
Expected Shortfall (ES) using the parametric method, assuming normally
distributed returns.
"""
import numpy as np
import scipy.stats

# Changed from relative import to absolute import from 'src' package
# This assumes the project root (parent of 'src') is in sys.path
# when this module is run or imported.
from src.utils import convert_annual_to_daily

def calculate_parametric_var_es(portfolio_config: dict):
    """
    Calculates VaR and ES using the Parametric (Variance-Covariance) method.

    Args:
        portfolio_config (dict): A dictionary containing all necessary parameters:
            - 'portfolio_value': float, total value of the portfolio.
            - 'weights': np.ndarray, asset weights.
            - 'expected_annual_returns': np.ndarray, annualized expected returns for assets.
            - 'annual_volatilities': np.ndarray, annualized volatilities for assets.
            - 'correlation_matrix': np.ndarray, correlation matrix of asset returns.
            - 'confidence_level': float, e.g., 0.99 for 99%.
            - 'time_horizon_days': int, VaR/ES time horizon in days.
            - 'trading_days_per_year': int, e.g., 252.

    Returns:
        tuple: (var_value, es_value, var_return, es_return)
               - var_value (float): Value-at-Risk as a positive loss value.
               - es_value (float): Expected Shortfall as a positive loss value.
               - var_return (float): VaR as a negative return (e.g., -0.02 for -2%).
               - es_return (float): ES as a negative return.

    Assumptions:
        - Asset returns are normally distributed.
        - Portfolio returns are normally distributed.
        - Daily returns are independent and identically distributed (IID) for scaling.
        Formulas are based on standard financial risk management texts,
        e.g., "Handbook of Financial Risk Management" by T. Roncalli (2020).
    """
    # Unpack parameters from config for easier access
    portfolio_value = portfolio_config['portfolio_value']
    weights = portfolio_config['weights']
    annual_returns = portfolio_config['expected_annual_returns']
    annual_vols = portfolio_config['annual_volatilities']
    corr_matrix = portfolio_config['correlation_matrix']
    confidence_level = portfolio_config['confidence_level']
    time_horizon = portfolio_config['time_horizon_days']
    trading_days = portfolio_config['trading_days_per_year']

    alpha = 1 - confidence_level  # Tail probability

    # 1. Convert annual figures to daily
    # The convert_annual_to_daily function is imported from src.utils
    daily_returns = convert_annual_to_daily(annual_returns, trading_days, is_volatility=False)
    daily_vols = convert_annual_to_daily(annual_vols, trading_days, is_volatility=True)

    # 2. Calculate Portfolio Expected Daily Return
    # E[R_p_daily] = w^T * mu_daily
    portfolio_daily_mean_return = np.dot(weights, daily_returns)

    # 3. Calculate Portfolio Daily Covariance Matrix & Variance/Volatility
    # Daily Covariance Matrix (Sigma_daily):
    # D_daily = diag(daily_vols)
    # Sigma_daily = D_daily * Correlation_Matrix * D_daily
    D_daily = np.diag(daily_vols)
    daily_covariance_matrix = D_daily @ corr_matrix @ D_daily
    
    # Portfolio Daily Variance (sigma^2_p_daily):
    # sigma^2_p_daily = w^T * Sigma_daily * w
    portfolio_daily_variance = weights.T @ daily_covariance_matrix @ weights
    
    # Portfolio Daily Volatility (sigma_p_daily):
    # Ensure variance is not negative due to floating point issues before sqrt
    if portfolio_daily_variance < 0:
        # This case should ideally not happen with valid inputs (PD correlation matrix)
        # but as a safeguard for sqrt:
        portfolio_daily_variance = 0 
    portfolio_daily_volatility = np.sqrt(portfolio_daily_variance)

    # 4. Adjust for Time Horizon
    # Horizon-Adjusted Mean Return (E[R_p_T]): E[R_p_daily] * T
    adj_portfolio_mean_return = portfolio_daily_mean_return * time_horizon
    
    # Horizon-Adjusted Volatility (sigma_p_T): sigma_p_daily * sqrt(T)
    adj_portfolio_volatility = portfolio_daily_volatility * np.sqrt(time_horizon)

    # 5. Calculate Parametric VaR
    # Z_alpha is the alpha-quantile of the standard normal distribution
    z_score = scipy.stats.norm.ppf(alpha) # For left tail (losses), this will be negative

    # VaR (as a return): E[R_p_T] + sigma_p_T * Z_alpha
    # This gives the worst expected return at the given confidence level.
    var_parametric_return = adj_portfolio_mean_return + adj_portfolio_volatility * z_score
    
    # VaR (as a positive loss value): -VaR_return * Portfolio_Value
    var_parametric_value = -var_parametric_return * portfolio_value
    var_parametric_value = max(0, var_parametric_value)


    # 6. Calculate Parametric Expected Shortfall (ES)
    # ES_return(alpha) = E[R_p_T] - sigma_p_T * (phi(Z_alpha) / alpha)
    # where phi(Z_alpha) is the PDF of the standard normal distribution at Z_alpha.
    # Check for alpha being zero to prevent division by zero, though ppf would fail first for alpha=0 or 1
    if alpha == 0: # Should not happen with typical confidence levels
        es_parametric_return = adj_portfolio_mean_return - adj_portfolio_volatility * np.inf
    elif adj_portfolio_volatility == 0: # If volatility is zero, ES is just the negative mean return if it's a loss
        es_parametric_return = adj_portfolio_mean_return
    else:
        pdf_at_z_score = scipy.stats.norm.pdf(z_score)
        es_parametric_return = adj_portfolio_mean_return - adj_portfolio_volatility * (pdf_at_z_score / alpha)
    
    # ES (as a positive loss value): -ES_return * Portfolio_Value
    es_parametric_value = -es_parametric_return * portfolio_value
    es_parametric_value = max(0, es_parametric_value) # Ensure non-negative loss

    return var_parametric_value, es_parametric_value, var_parametric_return, es_parametric_return

if __name__ == '__main__':
    # This block allows testing this module directly.
    # The sys.path manipulation below is to help Python find the 'src' package
    # when this script is run directly, enabling the `from src.utils import ...`
    # style imports for testing purposes within this __main__ block.
    import sys
    import os
    
    # Get the absolute path of the current file's directory (src)
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the absolute path of the parent directory (project root)
    project_root_dir = os.path.dirname(current_file_dir)
    
    # Add the project root directory to sys.path if it's not already there
    # This allows imports like `from src.module import ...`
    if project_root_dir not in sys.path:
        sys.path.insert(0, project_root_dir)

    # The top-level import `from src.utils import convert_annual_to_daily` should work
    # if the project root is in sys.path (which the above lines try to ensure for direct execution).
    # If it still fails, it means the execution environment for "<string>" doesn't pick this up.
    
    
    _convert_annual_to_daily_for_test = None
    try:
        # Try to use the already imported version if available
        _convert_annual_to_daily_for_test = convert_annual_to_daily
    except NameError: # Should not happen if top-level import worked
        try:
            from src.utils import convert_annual_to_daily as util_convert_main
            _convert_annual_to_daily_for_test = util_convert_main
        except ImportError:
            print("Warning: Could not import convert_annual_to_daily from src.utils for __main__ test.")
            print("Define a local version or ensure PYTHONPATH is set correctly for direct execution testing.")
            def local_dummy_convert(annual_value, trading_days, is_volatility=False):
                if is_volatility:
                    return annual_value / np.sqrt(trading_days)
                return annual_value / trading_days
            _convert_annual_to_daily_for_test = local_dummy_convert


    print("Testing Parametric VaR/ES Calculation Module...")
    sample_portfolio_for_test = {
        "name": "Parametric Test Portfolio",
        "portfolio_value": 1_000_000,
        "weights": np.array([0.6, 0.4]),
        "expected_annual_returns": np.array([0.10, 0.05]), # 10%, 5%
        "annual_volatilities": np.array([0.20, 0.10]),   # 20%, 10%
        "correlation_matrix": np.array([
            [1.0, 0.3],
            [0.3, 1.0]
        ]),
        "confidence_level": 0.99,
        "time_horizon_days": 10,
        "trading_days_per_year": 252,
        "num_simulations": 1000 # Not used by parametric
    }

    # The calculate_parametric_var_es function uses the `convert_annual_to_daily`
    # imported at the top of the file. The _convert_annual_to_daily_for_test
    # is primarily for ensuring the __main__ block itself can access it if needed
    # for setup, though in this specific structure, it's not directly passed or used by it.

    var_val, es_val, var_ret, es_ret = calculate_parametric_var_es(sample_portfolio_for_test)

    # For direct testing, we can use a simplified display
    print(f"\n--- Parametric Method Test Results ---")
    print(f"Portfolio Value: ${sample_portfolio_for_test['portfolio_value']:,.2f}")
    print(f"Confidence Level: {sample_portfolio_for_test['confidence_level']*100:.1f}%")
    print(f"Time Horizon: {sample_portfolio_for_test['time_horizon_days']} days")
    print("-" * 30)
    print(f"VaR Return: {var_ret:,.4%}")
    print(f"VaR Value: ${var_val:,.2f}")
    print(f"ES Return: {es_ret:,.4%}")
    print(f"ES Value: ${es_val:,.2f}")
    print("-" * 30)
