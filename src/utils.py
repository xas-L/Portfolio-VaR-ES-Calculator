"""
Utility Functions

This module contains helper functions for the PortfolioRiskCalculator,
such as display formatting, input validation (if needed), etc.
"""
import numpy as np

def display_results(method_name: str,
                    var_value: float,
                    es_value: float,
                    var_return: float,
                    es_return: float,
                    portfolio_config: dict):
    """
    Displays the calculated VaR and ES results in a formatted way.

    Args:
        method_name (str): Name of the calculation method (e.g., "Parametric").
        var_value (float): Calculated Value-at-Risk as a positive loss value.
        es_value (float): Calculated Expected Shortfall as a positive loss value.
        var_return (float): Calculated VaR as a negative return.
        es_return (float): Calculated ES as a negative return.
        portfolio_config (dict): The portfolio configuration dictionary used for calculation.
    """
    print(f"\n--- {method_name} Results ---")
    print(f"Portfolio: {portfolio_config.get('name', 'N/A')}")
    print(f"Initial Portfolio Value: ${portfolio_config.get('portfolio_value', 0):,.2f}")
    print(f"Confidence Level: {portfolio_config.get('confidence_level', 0)*100:.1f}%")
    print(f"Time Horizon: {portfolio_config.get('time_horizon_days', 0)} days")
    print("-" * 30)
    print(f"VaR ({portfolio_config.get('confidence_level', 0)*100:.1f}%) Return: {var_return:,.4%}")
    print(f"VaR ({portfolio_config.get('confidence_level', 0)*100:.1f}%) Value: ${var_value:,.2f}")
    print(f"ES ({portfolio_config.get('confidence_level', 0)*100:.1f}%) Return: {es_return:,.4%}")
    print(f"ES ({portfolio_config.get('confidence_level', 0)*100:.1f}%) Value: ${es_value:,.2f}")
    print("-" * 30)


def convert_annual_to_daily(annual_value: float, trading_days: int, is_volatility: bool = False) -> float:
    """
    Converts an annualized figure (return or volatility) to a daily figure.

    Args:
        annual_value (float): The annualized value (e.g., 0.08 for 8%).
        trading_days (int): The number of trading days in a year (e.g., 252).
        is_volatility (bool): True if the value is volatility (requires sqrt scaling),
                              False if it's a return (linear scaling).

    Returns:
        float: The corresponding daily value.
    """
    if is_volatility:
        return annual_value / np.sqrt(trading_days)
    else:
        return annual_value / trading_days

# You can add any other utility functions here, for example:
# def validate_portfolio_config(config):
#     """ Validates the structure and content of a portfolio configuration. """
#     # Implementation would go here
#     pass

if __name__ == '__main__':
    # Example usage of display_results (won't have real data yet)
    sample_config = {
        "name": "Test Display Portfolio",
        "portfolio_value": 500000,
        "confidence_level": 0.95,
        "time_horizon_days": 5
    }
    print("Testing display_results function (with dummy data):")
    display_results(
        method_name="Dummy Method",
        var_value=25000.00,
        es_value=35000.00,
        var_return=-0.05,
        es_return=-0.07,
        portfolio_config=sample_config
    )

    print("\nTesting convert_annual_to_daily function:")
    annual_ret = 0.10 # 10%
    annual_vol = 0.20 # 20%
    days = 252
    daily_ret = convert_annual_to_daily(annual_ret, days, is_volatility=False)
    daily_vol = convert_annual_to_daily(annual_vol, days, is_volatility=True)
    print(f"Annual Return: {annual_ret:.2%}, Daily Return: {daily_ret:.6%}")
    print(f"Annual Volatility: {annual_vol:.2%}, Daily Volatility: {daily_vol:.6%}")
