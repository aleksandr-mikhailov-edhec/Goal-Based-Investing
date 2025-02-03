import pandas as pd
import numpy as np
from scipy.optimize import minimize


# Assuming the corrected SmartBeta class is defined as provided earlier
class SmartBeta:
    def __init__(self, stockPrices, scheme, bounds=None, TargetReturn=0.05): 

        """
        Initialize the SmartBeta optimizer.

        Parameters:
        - stockPrices (pd.DataFrame): Asset prices.
        - scheme (str): Investment scheme ('EW', 'MSR', 'GMV', 'DR', 'RP', 'MV').
        - bounds (tuple, optional): Bounds for asset weights. Defaults to (0,1) for all assets.
        - TargetReturn (float): Target return for Mean-Variance optimization
        """

        self.smartScheme = scheme
        self.prices = stockPrices
        self.mu = TargetReturn
        self.returns = self.prices.pct_change().dropna()
        self.covMat = self._cov()
        self.lbsmb = 0
        self.ubsmb = 1
        self.tol = 1e-6
        self.rf = 0
        self.n = len(stockPrices.columns)
        self.ER = self.returns.mean()  # Define Expected Returns

        # Set bounds, default to (0, 1) if not provided
        if bounds is None:
            self.bounds = [(self.lbsmb, self.ubsmb)] * self.n
        else:
            self.bounds = [(bounds[0], bounds[1])] * self.n

    def _cov(self):
        cov_matrix = self.returns.cov()  # Covariance matrix of returns
        return cov_matrix

    def Constraint_sum_weights(self, w):
        return 1 - sum(w)

    def RP(self, w):
        cov_mat = np.array(self.covMat)
        w = np.array(w)
        vol = np.sqrt(np.dot(w, np.dot(cov_mat, w)))
        marginal_contribution = np.dot(cov_mat, w.T) / vol
        r = (vol / w.size) - np.multiply(w, marginal_contribution.T)
        rp = np.dot(r, r.T)  # Objective function
        return rp

    def DR(self, w):
        cov_mat = np.array(self.covMat)
        w = np.array(w)
        vol = np.sqrt(np.dot(w, np.dot(cov_mat, w)))
        weighted_var = np.dot(w.T, np.diag(cov_mat))
        DI = weighted_var / vol
        return -DI

    def MV(self, w):
        cov_mat = np.array(self.covMat)
        w = np.array(w)
        MV = np.dot(np.dot(w, cov_mat), w.T)
        return MV

    def SR(self, w):
        cov_mat = np.array(self.covMat)
        w = np.array(w)
        mean = np.dot(self.ER, w.T)
        vol = np.sqrt(np.dot(w, np.dot(cov_mat, w.T)))
        SR = (mean - self.rf) / vol
        return -SR  # Negative for minimization

    def Function_SmartBeta(self):
        x_0 = np.ones(self.n) / self.n
        bndsa = self.bounds

        if self.smartScheme == "EW":
            B = np.ones(self.n) / self.n
        elif self.smartScheme in ['MSR', 'GMV', 'DR', 'RP']:
            if self.smartScheme == 'MSR':
                objective2 = self.SR
            elif self.smartScheme == 'GMV':
                objective2 = self.MV
            elif self.smartScheme == 'DR':
                objective2 = self.DR
            elif self.smartScheme == 'RP':
                objective2 = self.RP

            cons2 = {'type': 'eq', 'fun': self.Constraint_sum_weights}
            res = minimize(
                objective2, 
                x_0, 
                method='SLSQP', 
                bounds=bndsa, 
                tol=self.tol, 
                constraints=cons2
            )
            if not res.success:
                raise ValueError(f"Optimization failed: {res.message}")
            B = res.x
        elif self.smartScheme == 'MV':
            objective2 = self.MV

            consTR = (
                {'type': 'eq', 'fun': self.Constraint_sum_weights},  
                {'type': 'eq', 'fun': lambda x: np.dot(x, self.ER) - self.mu}
            )
            res = minimize(
                objective2, 
                x_0, 
                method='SLSQP', 
                bounds=bndsa, 
                tol=self.tol, 
                constraints=consTR
            )
            if not res.success:
                raise ValueError(f"Optimization failed: {res.message}")
            B = res.x
        else:
            raise ValueError(f"Unsupported scheme: {self.smartScheme}")

        return B