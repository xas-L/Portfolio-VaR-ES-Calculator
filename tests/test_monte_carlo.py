# tests/test_monte_carlo.py
"""
Unit tests for the Monte Carlo VaR/ES calculation module.
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
    #    Check if parent of CWD contains a 'src' directory. Muchas gracias aficion SIUUU

    paths_to_check = []
    
    # Candidate 1: Based on __file__
    try:
        current_file_abspath = os.path.abspath(__file__)
        # Assuming tests/test_monte_carlo.py, so two os.path.dirname to get to project root
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

# Now imports from src should work
from src.monte_carlo_method import calculate_monte_carlo_var_es
from src.utils import convert_annual_to_daily # For test setup

class TestMonteCarloMethod(unittest.TestCase):

    def setUp(self):
        """Set up parameters for Monte Carlo tests."""
        self.mc_config = {
            "name": "MC Unit Test Portfolio",
            "portfolio_value": 1_000_000,
            "weights": np.array([0.5, 0.3, 0.2]),
            "confidence_level": 0.95, # Using 95% for MC tests for broader tail
            "time_horizon_days": 5,
            "num_simulations": 2000,  # Keep low for fast tests, increase for accuracy checks
            "trading_days_per_year": 252, # Needed for daily conversion setup
            # These annual figures are for setting up daily inputs for the MC function
            "expected_annual_returns": np.array([0.08, 0.03, 0.05]),
            "annual_volatilities": np.array([0.15, 0.05, 0.18]),
            "correlation_matrix": np.array([
                [1.0, 0.2, 0.1],
                [0.2, 1.0, 0.05],
                [0.1, 0.05, 1.0]
            ])
        }

        # Prepare daily inputs for the MC function
        self.daily_returns = convert_annual_to_daily(
            self.mc_config['expected_annual_returns'],
            self.mc_config['trading_days_per_year'],
            is_volatility=False
        )
        self.daily_vols = convert_annual_to_daily(
            self.mc_config['annual_volatilities'],
            self.mc_config['trading_days_per_year'],
            is_volatility=True
        )
        D_daily = np.diag(self.daily_vols)
        self.daily_covariance_matrix = D_daily @ self.mc_config['correlation_matrix'] @ D_daily
        
        # Seed for reproducibility in tests
        np.random.seed(42)


    def test_calculation_runs(self):
        """Test that the MC calculation runs without errors."""
        try:
            calculate_monte_carlo_var_es(
                portfolio_config=self.mc_config,
                daily_returns=self.daily_returns,
                daily_vols=self.daily_vols, # Not strictly used if daily_cov_matrix is passed
                daily_covariance_matrix=self.daily_covariance_matrix
            )
        except Exception as e:
            self.fail(f"calculate_monte_carlo_var_es raised an exception: {e}")

    def test_var_greater_than_es_values_mc(self):
        """Test that VaR value is typically less than or equal to ES value in MC."""
        # VaR and ES are positive loss values. ES should be >= VaR.
        var_val, es_val, _, _, _ = calculate_monte_carlo_var_es(
            portfolio_config=self.mc_config,
            daily_returns=self.daily_returns,
            daily_vols=self.daily_vols,
            daily_covariance_matrix=self.daily_covariance_matrix
        )
        self.assertGreaterEqual(es_val, var_val, "MC ES value should be >= VaR value")

    def test_var_return_less_than_es_return_mc(self):
        """Test that VaR return is typically greater than or equal to ES return in MC."""
        _, _, var_ret, es_ret, _ = calculate_monte_carlo_var_es(
            portfolio_config=self.mc_config,
            daily_returns=self.daily_returns,
            daily_vols=self.daily_vols,
            daily_covariance_matrix=self.daily_covariance_matrix
        )
        self.assertLessEqual(es_ret, var_ret, "MC ES return should be <= VaR return")

    def test_output_shapes(self):
        """Test the shape of the returned array of all simulated returns."""
        _, _, _, _, all_sim_returns = calculate_monte_carlo_var_es(
            portfolio_config=self.mc_config,
            daily_returns=self.daily_returns,
            daily_vols=self.daily_vols,
            daily_covariance_matrix=self.daily_covariance_matrix
        )
        self.assertEqual(all_sim_returns.shape[0], self.mc_config['num_simulations'],
                         "Number of simulated returns should match num_simulations.")

    def test_cholesky_error_handling(self):
        """Test that Cholesky decomposition failure is handled (if non-PD matrix)."""
        non_pd_matrix = np.array([[1.0, 2.0], [2.0, 1.0]]) # Not positive definite
        
        # Temporarily use a config that would lead to this non_pd_matrix as daily_covariance_matrix
        # We need to ensure the number of assets matches the non_pd_matrix
        temp_config = self.mc_config.copy()
        temp_config["weights"] = np.array([0.5, 0.5]) # Match 2 assets for non_pd_matrix

        # Dummy daily returns for 2 assets
        dummy_daily_returns = np.array([0.0, 0.0])
        # Dummy daily vols for 2 assets (not directly used if daily_covariance_matrix is passed)
        dummy_daily_vols = np.array([0.01, 0.01])


        with self.assertRaises(ValueError, msg="Should raise ValueError for non-PD matrix"):
            calculate_monte_carlo_var_es(
                portfolio_config=temp_config, # Use modified config
                daily_returns=dummy_daily_returns, 
                daily_vols=dummy_daily_vols,   
                daily_covariance_matrix=non_pd_matrix
            )

    # More tests could include:
    # - Consistency checks (e.g., higher volatility input leads to higher VaR/ES, though MC noise exists).
    # - Testing with very small number of simulations (e.g., 1 or 2) to check edge cases in indexing.
    # - If you implement different simulation methods for asset paths (e.g., GBM), test those.

if __name__ == '__main__':
    # This allows running the tests directly using `python tests/test_monte_carlo.py`
    # The sys.path modification at the top of the file is crucial for this to work.
    unittest.main()
