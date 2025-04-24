import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf  # Ensure scikit-learn is installed

class SmartBeta:
    def __init__(self, stockPrices, scheme, bounds=None, TargetReturn=0.05, 
                 use_relative_enc_constraint=False, relative_enc_target=0.5,
                 use_ledoit_wolf=False):
        """
        Initialize the SmartBeta optimizer.
        
        Parameters:
        - stockPrices (pd.DataFrame): Asset prices.
        - scheme (str): Investment scheme ('EW', 'MSR', 'GMV', 'DR', 'RP', 'MV').
        - bounds (tuple, optional): Bounds for asset weights (default: (0,1) for all assets).
        - TargetReturn (float): Target return for Mean-Variance optimization.
        - use_relative_enc_constraint (bool): Whether to apply the relative ENC constraint.
        - relative_enc_target (float): Target for relative ENC (default 0.5).
        - use_ledoit_wolf (bool): Whether to apply Ledoit–Wolf shrinkage to the covariance matrix.
        """
        self.smartScheme = scheme
        self.prices = stockPrices
        self.mu = TargetReturn
        self.returns = self.prices.pct_change().dropna()
        self.use_ledoit_wolf = use_ledoit_wolf  # New flag for Ledoit–Wolf shrinkage
        self.covMat = self._cov()
        self.lbsmb = 0
        self.ubsmb = 1
        self.tol = 1e-6
        self.rf = 0
        self.n = len(stockPrices.columns)
        self.ER = self.returns.mean()  # Expected Returns

        # Set bounds for each asset weight (default: (0,1))
        if bounds is None:
            self.bounds = [(self.lbsmb, self.ubsmb)] * self.n
        else:
            self.bounds = [(bounds[0], bounds[1])] * self.n
        
        self.use_relative_enc_constraint = use_relative_enc_constraint
        self.relative_enc_target = relative_enc_target

    def _cov(self):
        """
        Compute the covariance matrix of asset returns.
        Applies Ledoit–Wolf shrinkage if enabled.
        """
        if self.use_ledoit_wolf:
            lw = LedoitWolf()
            lw.fit(self.returns.values)
            cov_matrix = pd.DataFrame(
                lw.covariance_, 
                index=self.returns.columns, 
                columns=self.returns.columns
            )
            return cov_matrix
        else:
            return self.returns.cov()

    def Constraint_sum_weights(self, w):
        # Ensure the sum of weights equals 1.
        return 1 - np.sum(w)
    
    def constraint_relative_enc(self, w):
        """
        Enforces the relative ENC constraint:
        
        The relative effective number of constituents (ENC) is defined as:
            Relative ENC = 1/(N * sum(w_i^2))
        and we require:
            1/(N * sum(w_i^2)) - relative_enc_target = 0.
        """
        return 1/(self.n * np.sum(w**2)) - self.relative_enc_target

    def RP(self, w):
        cov_mat = np.array(self.covMat)
        w = np.array(w)
        vol = np.sqrt(np.dot(w, np.dot(cov_mat, w)))
        marginal_contribution = np.dot(cov_mat, w.T) / vol
        r = (vol / w.size) - np.multiply(w, marginal_contribution.T)
        rp = np.dot(r, r.T)
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
        return np.dot(np.dot(w, cov_mat), w.T)

    def SR(self, w):
        cov_mat = np.array(self.covMat)
        w = np.array(w)
        mean = np.dot(self.ER, w.T)
        vol = np.sqrt(np.dot(w, np.dot(cov_mat, w.T)))
        sr = (mean - self.rf) / vol
        return -sr  # Negative for minimization

    def Function_SmartBeta(self):
        """
        Compute portfolio weights based on the selected scheme and constraints.
        """
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

            # Base constraint: weights must sum to 1.
            constraints = [{'type': 'eq', 'fun': self.Constraint_sum_weights}]
            # Append the relative ENC constraint if enabled.
            if self.use_relative_enc_constraint:
                constraints.append({'type': 'eq', 'fun': self.constraint_relative_enc})
            
            res = minimize(
                objective2, 
                x_0, 
                method='SLSQP', 
                bounds=bndsa, 
                tol=self.tol, 
                constraints=constraints
            )
            if not res.success:
                raise ValueError(f"Optimization failed: {res.message}")
            B = res.x
        elif self.smartScheme == 'MV':
            objective2 = self.MV
            constraints = [
                {'type': 'eq', 'fun': self.Constraint_sum_weights},
                {'type': 'eq', 'fun': lambda x: np.dot(x, self.ER) - self.mu}
            ]
            if self.use_relative_enc_constraint:
                constraints.append({'type': 'eq', 'fun': self.constraint_relative_enc})
            
            res = minimize(
                objective2, 
                x_0, 
                method='SLSQP', 
                bounds=bndsa, 
                tol=self.tol, 
                constraints=constraints
            )
            if not res.success:
                raise ValueError(f"Optimization failed: {res.message}")
            B = res.x
        else:
            raise ValueError(f"Unsupported scheme: {self.smartScheme}")

        return B