# This file contains functions developed in other notebooks

import numpy as np
import pandas as pd

def market_simulation(
    # Basic notations
    T,        # Total time in years for the simulation
    N,        # Number of discrete time steps; e.g., 252 for "trading days" in a year
    M,        # Number of simulation paths (scenarios)
    seed=123, # Random seed for reproducibility
    
    # Initial conditions
    S0=1,         # Initial stock price
    r0=0.0306,    # Initial short rate
    sr0=0.4,      # Initial Sharpe ratio
    v0=(0.2138)**2, # Initial variance (square of initial volatility)
    
    # Short-rate (Vasicek) model parameters
    #lt_r=0.0306,    # Long-term mean (b in Vasicek) to which r reverts
    #kappa_r=0.13,   # Mean-reversion speed for short rate
    #sigma_r=0.98/100, # Volatility (std dev) of short rate’s diffusion
    
    # Sharpe-ratio (Ornstein–Uhlenbeck type) model parameters
    lt_sr=0.4,     # Long-term mean Sharpe ratio
    kappa_sr=0.35, # Mean-reversion speed for Sharpe ratio
    sigma_sr=0.2322, # Volatility (std dev) of Sharpe ratio process
    
    # Variance (Heston-like) model parameters
    kappa_variance=5.07,  # Mean-reversion speed of variance process
    lt_variance=(0.2138)**2, # Long-term mean of variance process
    sigma_variance=0.48,  # Vol of vol (volatility of variance process)
    
    # Correlations for covariance matrix
    # (Z is 4D, representing Brownian increments for: stock, variance, Sharpe ratio, short rate)
    rho_stock_volatility=-0.767, # Corr( dW_stock, dW_vol ) 
    rho_stock_sr=-0.2,           # Corr( dW_stock, dW_sr   )
    rho_volatility_sr=+0.767          # Corr( dW_vol,   dW_sr   )
):
    """
    Simulates multiple paths (M scenarios) of four stochastic processes over (N+1) time points:
      1. Stock price (S) following a log-Euler scheme with drift = (r + SR*sqrt(v) - 0.5*v)
      2. Variance (v) in a mean-reverting "Heston-like" process
      3. Sharpe ratio (sr) in an OU-style mean-reverting process
      4. Short rate (r) in a standard Vasicek model

    Underlying time steps: T (years) / N steps => dt = T/N

    The correlations among the Brownian increments are given by 'cov'.

    Parameters
    ----------
    T : float
        Total simulation horizon in years.
    N : int
        Number of discrete time steps.
    M : int
        Number of simulation paths/scenarios.
    seed : int, optional
        Random seed for reproducibility. Default = 123.
    S0 : float, optional
        Initial stock price.
    r0 : float, optional
        Initial short rate.
    sr0 : float, optional
        Initial Sharpe ratio.
    v0 : float, optional
        Initial variance. (Square of initial volatility)
    lt_r : float, optional
        Long-term mean for Vasicek short rate.
    kappa_r : float, optional
        Mean-reversion speed for short rate.
    sigma_r : float, optional
        Volatility parameter for short rate.
    lt_sr : float, optional
        Long-term mean for Sharpe ratio.
    kappa_sr : float, optional
        Mean-reversion speed for Sharpe ratio.
    sigma_sr : float, optional
        Volatility parameter for Sharpe ratio.
    kappa_variance : float, optional
        Mean-reversion speed for the variance process.
    lt_variance : float, optional
        Long-term mean for the variance process.
    sigma_variance : float, optional
        Volatility of the variance process (aka vol-of-vol).
    rho_stock_volatility : float, optional
        Correlation between the stock’s Brownian increment and the variance’s Brownian increment.
    rho_stock_sr : float, optional
        Correlation between the stock’s Brownian increment and the Sharpe ratio’s Brownian increment.
    rho_volatility_sr : float, optional
        Correlation between the variance’s Brownian increment and the Sharpe ratio’s Brownian increment.

    Returns
    -------
    (S, v, sr, r) : tuple of np.ndarray
        Each is an array of shape ((N+1), M):
            S[i, :] -> stock price at time step i for each of the M paths
            v[i, :] -> variance at time step i
            sr[i, :] -> Sharpe ratio at time step i
            r[i, :] -> short rate at time step i

    Notes
    -----
    - The code uses an exponential Euler scheme for the stock price: 
        S_{t+1} = S_t * exp( (r_t + sr_t * sqrt(v_t) - 0.5*v_t)*dt + sqrt(v_t*dt)*Z_{t,0} )
      This is a mix of a risk-free rate + Sharpe ratio drift approach. 
      (Under a strict risk-neutral Heston, you'd typically have just r_t - 0.5*v_t).
    - The variance v follows a mean-reverting square-root process (though not exactly the classic Heston if we omit the sqrt(v) in the drift).
    - The Sharpe ratio sr is an OU process with speed kappa_sr, mean lt_sr, vol sigma_sr.
    - The short rate r is a Vasicek process with speed kappa_r, mean lt_r, vol sigma_r.
    - The correlation matrix has 4 dimensions: [dW_stock, dW_vol, dW_sr, dW_r].
      We set Corr( dW_r, anything ) = 0 here. 
      If the final matrix is not positive semi-definite, you may get a runtime warning or numeric issues.

    Example usage:
    -------------
    S, v, sr, r = market_simulation(
        T=1.0, N=252, M=10000, seed=42,
        rho_stock_volatility=-0.7, rho_stock_sr=-0.2
    )
    """

    # Time step size in years
    dt = T / N
    
    # Mean (0-vector) and covariance matrix for Brownian increments
    mu = np.array([0, 0, 0, 0])
    cov = np.array([
        [1,                    rho_stock_volatility,  rho_stock_sr,     0],
        [rho_stock_volatility, 1,                     rho_volatility_sr,0],
        [rho_stock_sr,         rho_volatility_sr,     1,                0],
        [0,                    0,                     0,                1]
    ])

    # Initialize paths
    # (N+1) time points (including t=0), M scenarios
    S = np.full(shape=(N+1, M), fill_value=S0)  # stock
    v = np.full(shape=(N+1, M), fill_value=v0)  # variance
    sr = np.full(shape=(N+1, M), fill_value=sr0)# Sharpe ratio
    r = np.full(shape=(N+1, M), fill_value=r0)  # short rate
    
    # Draw correlated Brownian increments (Z) under the chosen measure
    # Z.shape = (N, M, 4) => For each time step, we have M sets of 4 correlated draws
    np.random.seed(seed)
    Z = np.random.multivariate_normal(mu, cov, size=(N, M))

    # Main simulation loop
    for i in range(1, N+1):
        # For notation convenience:
        #   i-1: previous time index
        #   Z[i-1,:,0]: Brownian increments for the stock
        #   Z[i-1,:,1]: Brownian increments for the variance
        #   Z[i-1,:,2]: Brownian increments for the Sharpe ratio
        #   Z[i-1,:,3]: Brownian increments for the short rate
        
        # -- Update Stock Price (exponential Euler) --
        S[i] = S[i-1] * np.exp(
            (r[i-1] + sr[i-1] * np.sqrt(v[i-1]) - 0.5*v[i-1]) * dt
            + np.sqrt(v[i-1] * dt) * Z[i-1,:,0]
        )
        
        # -- Update Variance (Heston-like mean reverting) --
        v[i] = (v[i-1]
                + kappa_variance * (lt_variance - v[i-1]) * dt
                + sigma_variance * np.sqrt(v[i-1]) * np.sqrt(dt) * Z[i-1,:,1])
        
        # Enforce non-negative variance
        v[i] = np.maximum(v[i], 0)
        
        # -- Update Sharpe Ratio (OU process) --
        sr[i] = (sr[i-1]
                 + kappa_sr*(lt_sr - sr[i-1]) * dt
                 + sigma_sr * np.sqrt(dt) * Z[i-1,:,2])

        # -- Update Short Rate (Vasicek) --
        #r[i] = (r[i-1]
        #        + kappa_r*(lt_r - r[i-1]) * dt
        #        + sigma_r * np.sqrt(dt) * Z[i-1,:,3])

    # Optional: Quick diagnostic check
    if np.sum(v < 0) > 0:
        print("Warning! Some variance values became negative (after truncation).")
    
    return S, v, sr#, r

def generate_equal_consumption_streams(
    accumulation_period_length=10,    # Number of years (periods) you are saving money
    cash_flows_accumulation=0,       # Annual cash flow during accumulation (e.g., how much is contributed/saved)
    decumulation_period_lenth=20,    # Number of years (periods) you are withdrawing money
    cash_flows_decumulation=-20000,  # Annual cash flow during decumulation (e.g., how much is withdrawn per year)
    inflation_rate=0.02              # Annual inflation rate (2% by default)
    ):
    """
    Generates a DataFrame of cash flows across accumulation and decumulation periods,
    including inflation adjustments.

    Returns:
        pandas.DataFrame with columns:
            - 'pure_CFs': The nominal cash flows (without considering inflation)
            - 'accumulated_inflation': The inflation growth factor over each period
            - 'inflation_adjusted_CFs': The real value of the cash flow in today's dollars
    """

    # Create a single array of cash flows:
    #   - 'accumulation_period_length' times 'cash_flows_accumulation' for the saving phase
    #   - 'decumulation_period_lenth' times 'cash_flows_decumulation' for the withdrawal phase
    cash_flows = np.array(
        [cash_flows_accumulation] * accumulation_period_length +
        [cash_flows_decumulation] * decumulation_period_lenth
    )

    # Create a time index (1 to total number of periods)
    dates = np.arange(1, accumulation_period_length + decumulation_period_lenth + 1, 1)
    
    # Build a DataFrame with a column 'pure_CFs' representing the nominal cash flows
    cash_flows_df = pd.DataFrame(cash_flows, columns=['pure_CFs'], index=dates)

    # For each period, calculate the accumulated inflation growth factor:
    # (1 + inflation_rate) ^ date_index
    cash_flows_df['accumulated_inflation'] = (inflation_rate + 1) ** dates

    # Multiply nominal cash flows by the inflation factor to get inflation-adjusted flows
    cash_flows_df['inflation_adjusted_CFs'] = (
        cash_flows_df['accumulated_inflation'] * cash_flows_df['pure_CFs']
    )

    return cash_flows_df