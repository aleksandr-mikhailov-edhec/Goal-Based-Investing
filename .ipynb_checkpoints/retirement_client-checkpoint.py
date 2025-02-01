import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class RetirementClient:
    def __init__(self,
                 name,
                 accumulation_years=10,
                 accumulation_cash_flow=0,
                 decumulation_years=20,
                 decumulation_cash_flow=50000,
                 periodicity = 12
                ):
        """
        Initializes a RetirementClient with given parameters.

        :param name: Name of the client
        :param accumulation_years: Number of years the client is saving money
        :param accumulation_cash_flow: Annual contribution during accumulation phase (per trading day)
        :param decumulation_years: Number of years the client is withdrawing money
        :param decumulation_cash_flow: Annual withdrawal amount during decumulation phase (only at start of each year)
        :param periodicity: Number of periods in a year
        """
        self.name = name
        self.accumulation_years = accumulation_years
        self.accumulation_cash_flow = accumulation_cash_flow
        self.decumulation_years = decumulation_years
        self.decumulation_cash_flow = decumulation_cash_flow
        self.periodicity = periodicity
        self.generate_cash_flows()
        
    def generate_cash_flows(self):
        """
        Generates a DataFrame of cash flows at a daily trading frequency (1/252 increments).
        
        - Accumulation period (X years, X*periodicity periods): Cash flow appears at period 1 of each year.
        - Decumulation period (Y years, Y*periodicity periodss): Cash flow appears at period 1 of each year.

        :return: pandas DataFrame with the following columns:
            - 'Period': The time index in trading days (1/252 increments)
            - 'Year': The corresponding year number
            - 'Pure_Cash_Flow': The nominal cash flows (without inflation adjustments)
        """
        total_periods = (self.accumulation_years + self.decumulation_years) * self.periodicity  # Total periods
        periods = np.arange(0, total_periods + 1)  # period index
        
        # Initialize cash flow array with zeros
        cash_flows = np.zeros(total_periods+1)
        
        # Set cash flow at the beginning of each year during decumulation
        for year in range(self.accumulation_years, self.accumulation_years + self.decumulation_years):
            cash_flows[year * self.periodicity] = self.decumulation_cash_flow
        
        # Convert to DataFrame
        self.cash_flows_df = pd.DataFrame({
            'Period': periods / self.periodicity, 
            'Year': (periods // self.periodicity) + 1,
            'Pure_Cash_Flow': cash_flows
            })
        self.cash_flows_df.index = self.cash_flows_df.Period

    def plot_cash_flows(self):
        """
        Plots the cash flow over the trading years.
        """
        plt.figure(figsize=(12, 5))
        plt.bar(self.cash_flows_df['Period'], self.cash_flows_df['Pure_Cash_Flow'], 
                color='darkblue', edgecolor='white', width=0.1)
        plt.xlabel("Time (Trading Years)")
        plt.ylabel("Cash Flow")
        plt.title(f"Cash Flow Projection for {self.name}")
        plt.axhline(0, color='black', linewidth=1)
        plt.show()