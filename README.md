# Portfolio Risk Calculator

**A Python-based tool for calculating portfolio Value-at-Risk (VaR) and Expected Shortfall (ES) using Parametric (Variance-Covariance) and Monte Carlo simulation methods.**

This project implements key financial risk measures, drawing significantly from concepts outlined in Thierry Roncalli's "Handbook of Financial Risk Management" (Chapman & Hall/CRC, 2020). It aims to provide a practical understanding of risk assessment through clear code and established financial theory.

## Table of Contents

- [Introduction](#introduction)
- [Core Risk Management Concepts Applied](#core-risk-management-concepts-applied)
  - [Value-at-Risk (VaR)](#value-at-risk-var)
  - [Expected Shortfall (ES)](#expected-shortfall-es)
- [Methodologies Implemented](#methodologies-implemented)
  - [Parametric (Variance-Covariance) Method](#parametric-variance-covariance-method)
  - [Monte Carlo Simulation Method](#monte-carlo-simulation-method)
- [Project Structure](#project-structure)
- [Python File Walkthrough](#python-file-walkthrough)
  - [`src/main.py`](#srcmainpy)
  - [`src/parametric_method.py`](#srcparametric_methodpy)
  - [`src/monte_carlo_method.py`](#srcmonte_carlo_methodpy)
  - [`src/utils.py`](#srcutilspy)
  - [`config/portfolio_config.py` (Conceptual - Embedded in `main.py`)](#configportfolio_configpy-conceptual---embedded-in-mainpy)
  - [`tests/test_parametric.py` and `tests/test_monte_carlo.py`](#teststest_parametricpy-and-teststest_monte_carlopy)
- [Setup and Installation](#setup-and-installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Running the Calculator](#running-the-calculator)
  - [Example Output](#example-output)
- [Testing](#testing)
- [Dependencies](#dependencies)
- [References](#references)
- [License](#license)

## Introduction

The Portfolio Risk Calculator project provides a Python-based implementation for quantifying financial risk in investment portfolios. Its primary objective is to calculate two critical risk metrics: Value-at-Risk (VaR) and Expected Shortfall (ES). By employing both the Parametric (Variance-Covariance) approach and Monte Carlo simulation, the tool offers different perspectives on potential portfolio losses.

The theoretical foundations for this project are rooted in established financial risk management literature, with a particular reliance on the "Handbook of Financial Risk Management" by Thierry Roncalli (2020), especially concepts from Chapter 2 regarding VaR and ES calculations. This project emphasizes modular code, clarity in implementation, and the practical application of financial theory.

## Core Risk Management Concepts Applied

The project centers on calculating and understanding two key risk measures:

### Value-at-Risk (VaR)

Value-at-Risk (VaR) is a statistical measure that quantifies the maximum expected loss over a specified time horizon at a given confidence level, under normal market conditions. For instance, a 1-day 99% VaR of \$100,000 means there is a 1% chance that the portfolio will lose more than \$100,000 over the next day. VaR is calculated both as an absolute monetary value and as a percentage return.

### Expected Shortfall (ES)

Expected Shortfall (ES), also known as Conditional VaR (CVaR) or Average Value-at-Risk (AVaR), measures the expected loss in the tail of the portfolio's return distribution, specifically when losses exceed the VaR threshold. ES provides a more comprehensive view of tail risk by answering: "If losses exceed VaR, what is the average magnitude of that loss?" ES is generally considered a more coherent risk measure than VaR because it better captures the severity of extreme losses.

These concepts are fundamental to contemporary risk management practices and are thoroughly detailed in Roncalli (2020).

## Methodologies Implemented

The calculator employs two distinct methodologies for VaR and ES estimation:

### Parametric (Variance-Covariance) Method

This approach assumes that individual asset returns are normally distributed, which implies that the overall portfolio return is also normally distributed.

* **Assumptions:**
    * Normality of asset and portfolio returns.
    * Daily returns are Independent and Identically Distributed (IID), allowing for time-scaling of risk metrics (mean scales with time $T$, volatility scales with $\sqrt{T}$).
* **Process:**
    1.  **Portfolio Statistics:** Calculate the portfolio's expected daily return ($E[R_p]$) and daily volatility ($\sigma_p$). This involves using individual asset weights ($w_i$), their expected daily returns ($\mu_i$), daily volatilities ($\sigma_i$), and the asset correlation matrix ($\rho_{ij}$). The portfolio's daily variance is computed as $\sigma^2_{p,daily} = \mathbf{w}^T \Sigma_{daily} \mathbf{w}$, where $\Sigma_{daily}$ is the daily covariance matrix of asset returns.
    2.  **Time Scaling:** Adjust the daily mean return and daily volatility to the desired time horizon $T$.
    3.  **VaR Calculation:** VaR is determined using the Z-score ($Z_{\alpha}$) corresponding to the desired confidence level ($1-\alpha$) from the standard normal distribution:
        $$ \text{VaR}_{\text{return}}(\alpha) = E[R_{p,T}] + \sigma_{p,T} \times Z_{\alpha} $$
    4.  **ES Calculation:** ES is calculated using the formula that incorporates the probability density function (PDF, $\phi(\cdot)$) of the standard normal distribution at the VaR Z-score:
        $$ ES_{\text{return}}(\alpha) = E[R_{p,T}] - \sigma_{p,T} \frac{\phi(Z_{\alpha})}{\alpha} $$
    These formulas are consistent with those presented in Roncalli (2020, Chapter 2).

### Monte Carlo Simulation Method

This method leverages computational power to simulate a large number of possible future return paths for the portfolio, generating an empirical distribution from which VaR and ES are estimated.

* **Assumptions:**
    * Asset returns can be modeled by a multivariate normal distribution (though other distributions could be incorporated).
    * Correlated asset returns are generated using Cholesky decomposition of their covariance matrix.
    * Daily returns are simulated and then compounded over the specified time horizon.
* **Process:**
    1.  **Covariance & Cholesky:** Calculate the daily covariance matrix ($\Sigma_{daily}$) of asset returns. Perform Cholesky decomposition to find a matrix $L$ such that $LL^T = \Sigma_{daily}$. This $L$ matrix is used to generate correlated random numbers.
    2.  **Simulation Loop:** For a large number of simulation paths ($N_{sim}$):
        * For each day within the defined time horizon $T$:
            * Generate a vector of independent standard normal random numbers ($Z_{day}$).
            * Simulate correlated daily asset returns: $\Delta R_{assets,day} = \text{expected\_daily\_returns} + L \cdot Z_{day}$.
            * Calculate the portfolio's aggregate daily return based on asset weights.
            * Compound the portfolio's value (or track its cumulative return) for that day.
        * Record the portfolio's final return over the entire horizon for the current simulation path.
    3.  **Risk Metric Derivation:** After all simulations are complete:
        * Sort the $N_{sim}$ simulated portfolio horizon returns in ascending order.
        * VaR is identified as the return at the $\alpha$-th percentile of this sorted empirical distribution.
        * ES is calculated as the average of all simulated returns that are worse than (i.e., less than or equal to) the VaR return.

## Project Structure

The project is organized with the following directory structure:

Portfolio-VaR-ES-Calculator/├── .git/├── .gitignore├── README.md├── requirements.txt├── LICENSE                   # (Optional: e.g., MIT License file)│├── config/                   # (Conceptual: Configuration currently in src/main.py)│   └── portfolio_config.py│├── docs/                     # (Optional: For supplementary documentation like ProjectSummary.pdf)│├── src/                      # Source code│   ├── init.py│   ├── main.py               # Main script, defines portfolio, orchestrates calculations│   ├── parametric_method.py  # Parametric VaR & ES logic│   ├── monte_carlo_method.py # Monte Carlo VaR & ES logic│   └── utils.py              # Utility functions (display, data conversion)│└── tests/                    # Unit tests├── init.py├── test_parametric.py└── test_monte_carlo.py
## Python File Walkthrough

### `src/main.py`

* **Purpose:** This script is the central orchestrator of the risk calculations.
* **Functionality:**
    * Defines the `DEFAULT_PORTFOLIO` configuration (asset details, weights, market assumptions, risk parameters).
    * Prepares daily financial figures (expected returns, volatilities, covariance matrix) from annualized inputs using helper functions from `utils.py`.
    * Calls `calculate_parametric_var_es()` from `parametric_method.py` to get Parametric VaR/ES.
    * Calls `calculate_monte_carlo_var_es()` from `monte_carlo_method.py` to get Monte Carlo VaR/ES.
    * Uses `display_results()` from `utils.py` to present the calculated risk metrics in a user-friendly format.
    * Optionally, generates and displays a histogram of the simulated portfolio returns from the Monte Carlo method using `matplotlib`.
* **Risk Concept Application:** Manages the overall workflow, ensuring that portfolio data and risk parameters are correctly fed into the respective calculation modules.

### `src/parametric_method.py`

* **Purpose:** Contains the implementation of the Parametric (Variance-Covariance) method for VaR and ES.
* **Key Function:** `calculate_parametric_var_es(portfolio_config)`
    * Accepts the portfolio configuration dictionary.
    * Converts annualized return and volatility inputs to daily figures.
    * Calculates the portfolio's daily mean return and daily variance (using matrix algebra: $\mathbf{w}^T \Sigma_{daily} \mathbf{w}$), then its daily volatility.
    * Scales these daily figures to the specified risk horizon.
    * Utilizes `scipy.stats.norm.ppf()` to obtain the Z-score for VaR and `scipy.stats.norm.pdf()` for the probability density value required for ES.
    * Returns VaR and ES as both absolute monetary values and percentage returns.
* **Risk Concept Application:** Directly applies the mathematical formulas for VaR and ES under the assumption of normally distributed returns, as detailed in financial literature (e.g., Roncalli, 2020, Chapter 2).

### `src/monte_carlo_method.py`

* **Purpose:** Implements the Monte Carlo simulation approach to estimate VaR and ES.
* **Key Function:** `calculate_monte_carlo_var_es(portfolio_config, daily_returns, daily_vols, daily_covariance_matrix)`
    * Takes the portfolio configuration and pre-calculated daily asset statistics as input.
    * Performs Cholesky decomposition (`np.linalg.cholesky()`) on the daily covariance matrix to facilitate the generation of correlated asset return paths.
    * Executes a loop for the specified number of simulations:
        * Within each simulation, iterates through each day of the risk horizon, generating correlated daily asset returns using the Cholesky factor and random numbers drawn from a standard normal distribution (`np.random.normal()`).
        * Calculates the portfolio's return for each day and compounds its value over the horizon.
    * After all simulations, sorts the resulting distribution of simulated end-of-horizon portfolio returns.
    * Determines VaR as the relevant percentile (e.g., the 1st percentile for a 99% confidence level) from this sorted distribution.
    * Calculates ES as the average of the simulated returns that fall in the tail beyond the VaR point.
    * Returns VaR and ES (values and returns), along with the full array of simulated returns for potential further analysis or plotting.
* **Risk Concept Application:** Implements a simulation-based estimation of risk. The derivation of VaR from the empirical distribution's percentile and ES as the conditional mean of tail losses are direct applications of their definitions.

### `src/utils.py`

* **Purpose:** A utility module containing helper functions to support the main calculation scripts and maintain code clarity.
* **Key Functions:**
    * `display_results(...)`: Formats and prints the calculated VaR and ES figures, along with key input parameters, for clear and understandable output.
    * `convert_annual_to_daily(...)`: Converts annualized financial figures (returns and volatilities) to their daily equivalents. Crucially, it correctly applies linear scaling for returns and square-root-of-time scaling for volatilities.
* **Risk Concept Application:** While not performing direct risk calculations, these functions are vital for correct data preparation (e.g., time scaling of parameters) and effective communication of the risk assessment results.

### `config/portfolio_config.py` (Conceptual - Embedded in `main.py`)

* **Purpose (Original Intent):** To provide a dedicated location for defining and storing various portfolio configurations, including asset allocations, market expectations (returns, volatilities, correlations), and risk parameters (confidence level, time horizon).
* **Current Implementation:** For operational simplicity and to avoid import issues in some execution environments, the primary `DEFAULT_PORTFOLIO` configuration is currently defined as a dictionary directly within `src/main.py`. The structure and content of this dictionary serve the same purpose as an external configuration file.
* **Risk Concept Application:** This component (whether as a separate file or embedded) is fundamental as it supplies all the necessary inputs—market views, portfolio structure, and risk preferences—that form the basis of the VaR and ES calculations. The quality and realism of these inputs significantly influence the meaningfulness of the resulting risk metrics.

### `tests/test_parametric.py` and `tests/test_monte_carlo.py`

* **Purpose:** These files contain unit tests designed to verify the correctness and reliability of the VaR and ES calculation logic within their respective modules. They utilize Python's built-in `unittest` framework.
* **Functionality:**
    * `test_parametric.py`: Includes tests to ensure the parametric calculations run without errors, to confirm the expected relationship between VaR and ES (ES representing a more severe or equal loss), and to validate results against known outcomes for simplified test cases (e.g., a single asset with zero mean return).
    * `test_monte_carlo.py`: Contains similar tests for the Monte Carlo method, including checks for basic execution, the VaR/ES relationship, the shape of output arrays (e.g., number of simulated returns), and error handling (such as for non-positive definite covariance matrices which would prevent Cholesky decomposition). `np.random.seed(42)` is used to ensure reproducibility of Monte Carlo test results.
* **Risk Concept Application:** The tests ensure that the Python code accurately implements the mathematical formulas and logical steps of the chosen VaR and ES methodologies. For example, testing the parametric VaR against a known Z-score for a simple scenario directly validates the core formula application.

## Setup and Installation

1.  **Prerequisites:**
    * Python 3.7 or higher.
    * Git (for cloning the repository).

2.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)<YOUR_USERNAME>/Portfolio-VaR-ES-Calculator.git
    cd Portfolio-VaR-ES-Calculator
    ```
    *(Replace `<YOUR_USERNAME>` with your actual GitHub username.)*

3.  **Create and Activate a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    ```
    * On Windows: `venv\Scripts\activate`
    * On macOS/Linux: `source venv/bin/activate`

4.  **Install Dependencies:**
    With the virtual environment activated, install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

The primary portfolio and risk parameters are defined within the `DEFAULT_PORTFOLIO` dictionary located at the beginning of the `src/main.py` script. You can modify this dictionary to customize the calculations:

* `name`: (str) A descriptive name for the portfolio.
* `portfolio_value`: (float) The total initial market value of the portfolio.
* `asset_names`: (list of str) Names of the assets in the portfolio.
* `weights`: (NumPy array) The proportion of the portfolio's total value allocated to each asset. Must sum to 1.0.
* `expected_annual_returns`: (NumPy array) The expected annualized return for each asset.
* `annual_volatilities`: (NumPy array) The expected annualized volatility (standard deviation of returns) for each asset.
* `correlation_matrix`: (NumPy array) A square matrix defining the correlation coefficients between the returns of each pair of assets.
* `confidence_level`: (float) The confidence level for VaR and ES calculations (e.g., `0.99` for 99%).
* `time_horizon_days`: (int) The time period (in days) over which the risk is to be estimated.
* `num_simulations`: (int) The number of simulation paths to run for the Monte Carlo method.
* `trading_days_per_year`: (int) The number of trading days assumed in a year (e.g., 252), used for converting annual figures to daily.

## Usage

### Running the Calculator

To execute the VaR and ES calculations, navigate to the root directory of the project in your terminal (the directory containing the `src` folder) and run the `main.py` script as a module:

```bash
python -m src.main
This command will:Load the DEFAULT_PORTFOLIO configuration from src/main.py.Perform VaR and ES calculations using both the Parametric and Monte Carlo methods.Print the detailed results to the console.If matplotlib is correctly installed and configured, it may display a histogram of the simulated portfolio returns from the Monte Carlo analysis (note: plt.show() is commented out in src/main.py to prevent blocking in non-interactive environments; uncomment if you want to see the plot interactively).Example OutputThe console output will resemble the following (Monte Carlo results will vary slightly due to randomness unless a seed is fixed globally in main.py):--- Starting Risk Calculations for: DEFAULT_PORTFOLIO ---

--- Parametric (Variance-Covariance) Results ---
Portfolio: Default Diversified Portfolio
Initial Portfolio Value: $1,000,000.00
Confidence Level: 99.0%
Time Horizon: 10 days
------------------------------
VaR (99.0%) Return: -X.XXXX%
VaR (99.0%) Value: $XX,XXX.XX
ES (99.0%) Return: -Y.YYYY%
ES (99.0%) Value: $YY,YYY.YY
------------------------------

Starting Monte Carlo simulation with 10000 paths...
(This may take a moment depending on the number of simulations and horizon)

--- Monte Carlo Simulation Results ---
Portfolio: Default Diversified Portfolio
Initial Portfolio Value: $1,000,000.00
Confidence Level: 99.0%
Time Horizon: 10 days
------------------------------
VaR (99.0%) Return: -A.AAAA%
VaR (99.0%) Value: $AA,AAA.AA
ES (99.0%) Return: -B.BBBB%
ES (99.0%) Value: $BB,BBB.BB
------------------------------
Plot generated (plt.show() is commented out for non-interactive environments).

--- Risk Calculations Finished ---
TestingThe project includes unit tests to verify the correctness of the calculation logic. These tests are located in the tests/ directory.To run all tests, navigate to the project's root directory and execute:python -m unittest discover tests
Alternatively, you can run individual test files:python tests/test_parametric.py
python tests/test_monte_carlo.py
The sys.path modifications at the beginning of the test files help ensure that modules from the src directory can be correctly imported when tests are run directly.DependenciesThe core dependencies for this project are listed in requirements.txt:NumPy: For efficient numerical computations, especially array and matrix operations.SciPy: Used for scientific and technical computing, particularly its statistical functions (scipy.stats.norm) for the Parametric method.Matplotlib: For generating plots, such as the histogram of simulated returns from the Monte Carlo method (optional for core calculation, but used for visualization in main.py).ReferencesRoncalli, T. (2020). Handbook of Financial Risk Management. Chapman & Hall/CRC Financial Mathematics Series. (Key reference, particularly Chapter 2 for VaR and ES definitions and