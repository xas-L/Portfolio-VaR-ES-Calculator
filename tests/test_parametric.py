"""
Unit tests for the Parametric VaR/ES calculation module.
"""
import unittest
import numpy as np
import sys
import os

# --- sys.path modification to allow imports from src ---
def _add_project_root_to_path_if_needed():
    # Strategy:
    # 1. Try to derive project root from __file__ (parent of parent of this file).
    #    Check if this derived root contains a 'src' directory.
    # 2. If not, try current working directory (os.getcwd()).
    #    Check if CWD contains a 'src' directory.
    # 3. If not, try parent of CWD.
    #    Check if parent of CWD contains a 'src' directory.

    paths_to_check = []
    
    # Candidate 1: Based on __file__
    try:
        current_file_abspath = os.path.abspath(__file__)
        # Assuming tests/test_parametric.py, so two os.path.dirname to get to project root
        project_root_from_file = os.path.dirname(os.path.dirname(current_file_abspath))
        paths_to_check.append(project_root_from_file)
    except NameError: # __file__ is not defined
        # print("DEBUG: __file__ not defined.", file=sys.stderr) # Optional debug
        pass
    except Exception as e:
        # print(f"DEBUG: Error determining path from __file__: {e}", file=sys.stderr) # Optional debug
        pass

    # Candidate 2: Current Working Directory
    cwd = os.getcwd()
    paths_to_check.append(cwd)

    # Candidate 3: Parent of Current Working Directory
    parent_of_cwd = os.path.dirname(cwd)
    if parent_of_cwd != cwd: # Avoid adding '.' if cwd is root, or adding same path twice
        if parent_of_cwd not in paths_to_check: # Avoid duplicate checks if CWD was already parent
             paths_to_check.append(parent_of_cwd)
    
    # Iterate through candidate paths and add the first one that contains 'src'
    for candidate_root in paths_to_check:
        # Ensure candidate_root is a valid string path before os.path.isdir
        if isinstance(candidate_root, str) and os.path.isdir(candidate_root) and \
           os.path.isdir(os.path.join(candidate_root, "src")):
            if candidate_root not in sys.path:
                sys.path.insert(0, candidate_root)
                # print(f"DEBUG: Added to sys.path: {candidate_root}", file=sys.stderr) # Optional debug
            return # Found and added

    # If no suitable path was found
    # print(f"Warning: Could not find project root containing 'src'. Current sys.path: {sys.path}, CWD: {cwd}", file=sys.stderr) # Optional debug

_add_project_root_to_path_if_needed()

# Now imports from the 'src' package should work.
from src.parametric_method import calculate_parametric_var_es
from src.utils import convert_annual_to_daily # For test setup if needed

class TestParametricMethod(unittest.TestCase):

    def setUp(self):
        """Set up a sample portfolio configuration for testing."""
        self.sample_portfolio_test = {
            "name": "Parametric Unit Test Portfolio",
            "portfolio_value": 1_000_000,
            "weights": np.array([0.6, 0.4]), # Two assets
            "expected_annual_returns": np.array([0.10, 0.05]), 
            "annual_volatilities": np.array([0.20, 0.10]),   
            "correlation_matrix": np.array([
                [1.0, 0.3],
                [0.3, 1.0]
            ]),
            "confidence_level": 0.99,
            "time_horizon_days": 1, # Using 1-day for simpler manual checks
            "trading_days_per_year": 252,
            "num_simulations": 1000 # Not used by parametric
        }

        # Pre-calculate some known values for a very simple case (single asset, zero mean)
        self.single_asset_zero_mean_config = {
            "name": "Single Asset Zero Mean Test",
            "portfolio_value": 1_000_000,
            "weights": np.array([1.0]),
            "expected_annual_returns": np.array([0.0]),
            "annual_volatilities": np.array([0.15873]), # Approx 1% daily vol (0.15873 / sqrt(252) ~= 0.01)
            "correlation_matrix": np.array([[1.0]]),
            "confidence_level": 0.99,
            "time_horizon_days": 1,
            "trading_days_per_year": 252,
        }


    def test_calculation_runs(self):
        """Test that the calculation runs without errors for a valid config."""
        try:
            calculate_parametric_var_es(self.sample_portfolio_test)
        except Exception as e:
            self.fail(f"calculate_parametric_var_es raised an exception: {e}")

    def test_var_greater_than_es_values(self):
        """Test that VaR value is typically less than or equal to ES value (ES is a worse loss)."""
        # Note: VaR and ES are positive loss values here. So ES should be >= VaR.
        var_val, es_val, _, _ = calculate_parametric_var_es(self.sample_portfolio_test)
        self.assertGreaterEqual(es_val, var_val, "ES value should be >= VaR value (representing a larger or equal loss)")

    def test_var_return_less_than_es_return(self):
        """Test that VaR return is typically greater than or equal to ES return (ES is more negative)."""
        _, _, var_ret, es_ret = calculate_parametric_var_es(self.sample_portfolio_test)
        # Both returns are negative, ES should be "more negative" or equal
        self.assertLessEqual(es_ret, var_ret, "ES return should be <= VaR return (more negative or equal)")


    def test_single_asset_zero_mean_var(self):
        """Test VaR for a single asset with zero mean return against known Z-score."""
        # For 1-day, 99% CL, zero mean, daily_vol = 0.01
        # Expected VaR_return = 0 + daily_vol * Z_alpha (Z_alpha for 0.01 is approx -2.3263)
        # Expected VaR_return = 0.01 * -2.3263 = -0.023263
        # Expected VaR_value = -(-0.023263) * 1_000_000 = 23263
        
        # Calculate daily vol for this specific config
        daily_vol = self.single_asset_zero_mean_config["annual_volatilities"][0] / np.sqrt(
            self.single_asset_zero_mean_config["trading_days_per_year"]
        ) # Should be ~0.01

        var_val, _, var_ret, _ = calculate_parametric_var_es(self.single_asset_zero_mean_config)
        
        expected_var_ret = daily_vol * (-2.3263478740408408) # scipy.stats.norm.ppf(0.01)
        expected_var_val = -expected_var_ret * self.single_asset_zero_mean_config["portfolio_value"]

        self.assertAlmostEqual(var_ret, expected_var_ret, places=6, msg="Single asset zero mean VaR return mismatch")
        self.assertAlmostEqual(var_val, expected_var_val, places=1, msg="Single asset zero mean VaR value mismatch")


    # Add more tests:
    # - Test with different confidence levels.
    # - Test with different time horizons.
    # - Test edge cases (e.g., zero volatility - though this might break some formulas if not handled).
    # - Test input validation if you add it to the main function.

if __name__ == '__main__':
    # This allows running the tests directly using `python tests/test_parametric.py`
    # The sys.path modification at the top of the file is crucial for this to work.
    unittest.main()
