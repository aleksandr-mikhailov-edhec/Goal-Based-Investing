import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

class MarketSimulator:
    def __init__(
        self, 
        # Basic notations
        T,        # Total time in years for the simulation
        N,        # Number of discrete time steps; e.g., 252 for "trading days" in a year
        M,        # Number of simulation paths (scenarios)
        seed=123, # Random seed for reproducibility
        
        # Initial conditions
        S0=1,         # Initial stock price
        r0=0.15 / 100,      # Initial short rate
        sr0=0.4,      # Initial Sharpe ratio
        v0=(0.2138)**2, # Initial variance (square of initial volatility)
        
        # Short-rate (Vasicek) model parameters
        lt_r=0.0306,    # Long-term mean (b in Vasicek) to which r reverts
        kappa_r=0.13,   # Mean-reversion speed for short rate
        sigma_r=0.98/100, # Volatility (std dev) of short rate’s diffusion
        lambda_r=-53/100/100, # Market price of risk associated with interest rate movements
        
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
        rho_volatility_sr=+0.767,    # Corr( dW_vol,   dW_sr   )

        # Constant maturity bond index set up
        tau = 3, # Time to maturity in years of the bond index (default = 3 years)
        B0 = 1   # Initial price of the bond index
    ):
        """Initializes the Market Simulator with all parameters."""
        
        # Simulation settings
        self.T, self.N, self.M, self.seed = T, N, M, seed  # Total time horizon, number of time steps, number of paths, random seed
        
        # Initial values
        self.S0, self.r0, self.sr0, self.v0 = S0, r0, sr0, v0  # Initial values for stock price, short rate, Sharpe ratio, and variance
        
        # Short-rate (Vasicek) model parameters
        self.lt_r, self.kappa_r, self.sigma_r, self.lambda_r = lt_r, kappa_r, sigma_r, lambda_r  # Long-term mean, mean reversion speed, volatility, and market price of risk
        
        # Sharpe-ratio (Ornstein–Uhlenbeck type) model parameters
        self.lt_sr, self.kappa_sr, self.sigma_sr = lt_sr, kappa_sr, sigma_sr  # Long-term mean, mean reversion speed, and volatility of Sharpe ratio
        
        # Variance (Heston-like) model parameters
        self.kappa_variance, self.lt_variance, self.sigma_variance = kappa_variance, lt_variance, sigma_variance  # Mean reversion speed, long-term mean, and vol of vol
        
        # Correlations for covariance matrix
        self.rho_stock_volatility, self.rho_stock_sr, self.rho_volatility_sr = rho_stock_volatility, rho_stock_sr, rho_volatility_sr  # Correlation among Brownian motions
        
        # Time step size in years
        self.dt = T / N  # Compute dt from total time and number of steps
        
        # Set up risk-neutral under Q measure long-term short rate
        self.lt_r_q = lt_r + sigma_r * lambda_r / kappa_r  # Adjusted long-term mean for risk-neutral valuation

        # generate the market paths
        self.generate_paths()

        # set up constant maturity bond index
        self.tau = tau 
        self.B0 = B0
        
        zero_coupon_bond_prices_for_constant_maturity_bond_index = self.vasicek_zcb_price(self.r_p, self.tau)
        
        # find price of constant maturity bond index
        self.B_p = self.deduce_constant_maturity_bond_index(zero_coupon_bond_prices_for_constant_maturity_bond_index)

        # set the retirement bond price to nothing. To initialize it run "self.calculate_perfect_retirement_bond(client)"
        # where client comes from 'RetirementClient' class
        self.retirement_bond_p = pd.DataFrame(index = self.r_p.index)
  
    def generate_paths(self):
        """Simulates multiple paths of stock price, variance, Sharpe ratio, and short rate."""
        np.random.seed(self.seed)  # Set random seed for reproducibility
        
        # Mean (0-vector) and covariance matrix for Brownian increments
        mu = np.array([0, 0, 0, 0])  # Zero mean for Brownian motion
        cov = np.array([
            [1, self.rho_stock_volatility, self.rho_stock_sr, 0],  # Correlation matrix for Brownian motions
            [self.rho_stock_volatility, 1, self.rho_volatility_sr, 0],
            [self.rho_stock_sr, self.rho_volatility_sr, 1, 0],
            [0, 0, 0, 1]  # Independent short rate process
        ])
        
        # Initialize paths
        S = np.full((self.N+1, self.M), self.S0)  # Stock price paths
        v = np.full((self.N+1, self.M), self.v0)  # Variance paths
        sr = np.full((self.N+1, self.M), self.sr0)  # Sharpe ratio paths
        r = np.full((self.N+1, self.M), self.r0)  # Short rate paths
        
        # Generate correlated Brownian increments
        Z = np.random.multivariate_normal(mu, cov, size=(self.N, self.M))
        
        # Main simulation loop
        for i in range(1, self.N+1):
            # Stock price evolution using exponential Euler scheme
            S[i] = S[i-1] * np.exp((r[i-1] + sr[i-1] * np.sqrt(v[i-1]) - 0.5 * v[i-1]) * self.dt + np.sqrt(v[i-1] * self.dt) * Z[i-1,:,0])
            
            # Variance follows a mean-reverting Heston-like process
            v[i] = np.maximum(v[i-1] + self.kappa_variance * (self.lt_variance - v[i-1]) * self.dt + self.sigma_variance * np.sqrt(v[i-1]) * np.sqrt(self.dt) * Z[i-1,:,1], 0)
            
            # Sharpe ratio follows an Ornstein–Uhlenbeck process
            sr[i] = sr[i-1] + self.kappa_sr * (self.lt_sr - sr[i-1]) * self.dt + self.sigma_sr * np.sqrt(self.dt) * Z[i-1,:,2]
            
            # Short rate follows a Vasicek model under the risk-neutral measure
            r[i] = r[i-1] + self.kappa_r * (self.lt_r_q - r[i-1]) * self.dt + self.sigma_r * np.sqrt(self.dt) * Z[i-1,:,3]
        
        # Convert outputs to DataFrame with time index

        self.S_p = pd.DataFrame(S, index=np.arange(0, self.T+self.dt-1e-10, self.dt))
        self.v_p = pd.DataFrame(v, index=np.arange(0, self.T+self.dt-1e-10, self.dt))
        self.sr_p = pd.DataFrame(sr, index=np.arange(0, self.T+self.dt-1e-10, self.dt))
        self.r_p = pd.DataFrame(r, index=np.arange(0, self.T+self.dt-1e-10, self.dt))

    def vasicek_zcb_price(self, r_t, tau):
        """
        Computes the price P(t, t+tau) of a zero-coupon bond under the Vasicek model.
    
        Parameters:
        - r_t: Short rate at time t
        - tau: Time to maturity in years (default = 3 years)
    
        Returns:
        - Price of the zero-coupon bond P(t, t+tau)
        """
        # Compute B(t, T) factor in Vasicek model
        B = (1.0 - np.exp(-self.kappa_r * tau)) / self.kappa_r
    
        # Compute A(t, T) factor in Vasicek model
        A_term1 = (self.lt_r_q - (self.sigma_r**2) / (2.0 * self.kappa_r**2)) * (B - tau)
        A_term2 = (self.sigma_r**2) / (4.0 * self.kappa_r) * B**2
        A = np.exp(A_term1 - A_term2)  # Fixed: Applied exponentiation as per formula
    
        # Compute bond price using Vasicek formula
        return A * np.exp(-B * r_t)

    def vasicek_zcb_yield(self, tau):
        """
        Computes the price P(t, t+tau) yeild curve of a zero-coupon bond under the Vasicek model.
    
        Parameters:
        - r_t: Short rate at time t
        - tau: Time to maturity in years (default = 3 years)
    
        Returns:
        - Yield curve of bond P(t, t+tau)
        """
        # Compute B(t, T) factor in Vasicek model
        B = (1.0 - np.exp(-self.kappa_r * self.tau)) / self.kappa_r
    
        # Compute A(t, T) factor in Vasicek model
        A_term1 = (self.lt_r_q - (self.sigma_r**2) / (2.0 * self.kappa_r**2)) * (B - tau)
        A_term2 = (self.sigma_r**2) / (4.0 * self.kappa_r) * B**2
        A = np.exp(A_term1 - A_term2)  # Fixed: Applied exponentiation as per formula
    
        # Compute bond price using Vasicek formula
        return -np.log(A * np.exp(-B * self.r_p))/tau
    
    def deduce_constant_maturity_bond_index(self, zero_coupon_prices):
        """
        Computes a constant maturity bond index based on zero-coupon bond prices and short-term interest rates.
    
        Parameters:
        zero_coupon_prices (DataFrame): Zero-coupon bond prices for different maturities over time.
        r_p (Series or DataFrame): Short-term interest rates (or bond yields).
        dt (float): Time step size.
        initial_investment (float, optional): Initial investment amount. Default is 100.
    
        Returns:
        DataFrame: Computed bond index values.
        """
        
        # Compute the return from changes in zero-coupon bond prices
        return_from_change_in_bond_price = zero_coupon_prices.pct_change().fillna(0)
    
        # Include the dt factor explicitly
        return_from_coupons = self.r_p * self.dt  # Keeps dt in formula
    
        # Compute total return index
        index_return = return_from_coupons + return_from_change_in_bond_price
    
        # Compute bond index, scaling by initial investment
        bond_index = self.B0 * (index_return+1).cumprod()
    
        return bond_index


    def calculate_perfect_retirement_bond(self, client):
        """
        This function calculates price of the perfect retirement bond 
        given the market conditions. This function uses parallel computing

        Essentially it calculates zc bond matrix for each scenario and 
        finds present value of the cash flows. these PVs are the perfect
        discounted values of CFs = perfect retirement bond. The least 
        risky asset that allows to reach the goal with certain probability

        Parameters:
        client (Class): A client initialized through the class "RetirementClient"
                        They should have views on their CFs + periodociy
                        periodicity is used for initializing cash flows more often
                        ### IMPORTANT ### Higher periodicity makes this code slower,

        Returns:
        df_pv_retirement_bonds (DataFrame): retirement bond price

        """
    
        # Initialize an empty DataFrame to store the results (retirement bond prices) for each simulation
        retirement_bond_p = pd.DataFrame(index=self.r_p.index)
        
        # This function handles the processing of one simulation at a time
        def process_simulation(simulation_number):
            """
            Processes a single simulation to calculate the present value (PV) of the retirement bond.
            The cash flows are discounted using the zero-coupon bond prices, which are calculated using
            the simulated short-term interest rates (r_p) for each period in the simulation.
    
            Parameters:
            simulation_number (int): The column index of the simulation to process.
            
            Returns:
            PVs (list): A list of the present values of the cash flows for the given simulation.
            """
            
            # Step 1: Initialization
            # CFs holds the client's cash flows dataframe. This will be used to calculate the discounted cash flows.
            CFs = client.cash_flows_df
            
            # r_p_short_term contains the simulated short-term interest rates for the current simulation
            r_p_short_term = self.r_p.loc[:, simulation_number]
            
            # periodicity_of_CF_shift is used to determine how often to shift the cash flows
            periodicity_of_CF_shift = 1 / client.periodicity
    
            # Step 2: ZC (Zero-Coupon) Bond Matrix Calculation
            # We calculate the zero-coupon bond prices for all periods (tau) for the current simulation.
            # These bond prices are calculated using the simulated short-term interest rates (r_p_short_term).
            zc_bond_prices_dict = {
                period: self.vasicek_zcb_price(r_p_short_term, tau=period) 
                for period in CFs.index
            }
            
            # Convert the dictionary of bond prices to a DataFrame for easy access
            zc_bond_prices = pd.DataFrame.from_dict(zc_bond_prices_dict, orient='index').T
    
            # Step 3: Calculate Present Value (PV) of Client's Cash Flows
            # PVs will hold the present values of all cash flows for the current simulation
            PVs = []
            current_period = 0
            
            # We loop over each period in the bond prices and calculate the discounted value of the cash flows
            for period in zc_bond_prices.index:
                # Discount the cash flows using the corresponding zero-coupon bond prices for the current period
                discounted_CFs = CFs['Pure_Cash_Flow'] * zc_bond_prices.loc[period, :]
                
                # Sum the discounted cash flows to get the total present value for this period
                PVs.append(np.sum(discounted_CFs))
        
                # Shift the cash flows for the next period, if necessary (every year based on periodicity)
                if current_period != period // periodicity_of_CF_shift:
                    current_period = period // periodicity_of_CF_shift
                    CFs = CFs.shift(-1).fillna(0)  # Shift cash flows for the next period
    
            # Return the list of present values for this simulation
            return PVs
    
        # Step 4: Parallelizing the Loop over All Simulations
        # Use joblib's Parallel and delayed functions to process all simulations in parallel
        # n_jobs=-1 uses all available CPU cores for parallel processing
        result = Parallel(n_jobs=-1)(delayed(process_simulation)(sim_num) for sim_num in tqdm(self.r_p.columns, desc="Simulating scenarios"))
        
        # Step 5: Save the Results
        # After all simulations are processed, we store the results in the DataFrame `retirement_bond_p`
        # Each column corresponds to a simulation, and each row corresponds to a period.
        for i, simulation_number in enumerate(self.r_p.columns):
            retirement_bond_p.loc[:, simulation_number] = result[i]
        
        # Save the final DataFrame with all simulation results
        self.retirement_bond_p = retirement_bond_p

    def plot_market_simulation(self):
        """
        This function plots the stock price, intereset rate,
        Stock volatility and stock sharpe ratio
        """
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))

        # --- (1) Heston Model log of Asset Prices ---
        axes[0, 0].plot(self.S_p.index, np.log(self.S_p), alpha=0.6)
        axes[0, 0].set_title("Heston Model Log of Asset Prices")
        axes[0, 0].set_xlabel("Time")
        axes[0, 0].set_ylabel("Log-Price Level")
        axes[0, 0].grid(True)
        
        # --- (2) Heston Model Variance Process ---
        axes[1, 0].plot(self.v_p.index, self.v_p, alpha=0.6)
        axes[1, 0].set_title("Heston Model Variance Process")
        axes[1, 0].set_xlabel("Time")
        axes[1, 0].set_ylabel("Variance")
        axes[1, 0].grid(True)
        
        # --- (3) Sharpe Ratio Process ---
        axes[0, 1].plot(self.sr_p.index, self.sr_p, alpha=0.6)
        axes[0, 1].set_title("Sharpe Ratio Process")
        axes[0, 1].set_xlabel("Time")
        axes[0, 1].set_ylabel("Sharpe Ratio")
        axes[0, 1].grid(True)
        
        # --- (4) Vasicek Process (Interest Rate) ---
        axes[1, 1].plot(self.r_p.index, self.r_p, alpha=0.6)
        axes[1, 1].set_title("Vasicek Process short term rate")
        axes[1, 1].set_xlabel("Time")
        axes[1, 1].set_ylabel("Interest Rate")
        axes[1, 1].grid(True)
        
        # --- (5) Constant Maturity Bond Index ---
        axes[2, 0].plot(self.B_p.index, self.B_p, alpha=0.6)
        axes[2, 0].set_title(f"Constant Maturity {self.tau} years Bond Index")
        axes[2, 0].set_xlabel("Time")
        axes[2, 0].set_ylabel("Price")
        axes[2, 0].grid(True)

        # --- (6) Retirement Bond ---
        axes[2, 1].plot(self.retirement_bond_p.index, self.retirement_bond_p, alpha=0.6)
        axes[2, 1].set_title(f"Perfect Retirement Bond Index")
        axes[2, 1].set_xlabel("Time")
        axes[2, 1].set_ylabel("Price")
        axes[2, 1].grid(True)
                
        plt.tight_layout()