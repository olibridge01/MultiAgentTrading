# min varaince portfolio allocation startegy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MinVariance:
    """
    Investment strategy that allocates capital to minimize the portfolio variance.
    Combines high-risk stocks to offset each other's risk.
    The global minimum variance portfolio is the portfolio at the leftmost point of the efficient frontier.
    """

    def __init__(self, returns: pd.DataFrame, initial_balance: float = 1000):
        self.n_assets = returns.shape[1]
        self.returns = returns

        self.allocation = None
        self.balance = initial_balance
        self.portfolio_value_history = [initial_balance]
        # self.cov_matrix = None
        # self.expected_returns = None

    # @property
    # def cov_matrix(self):
    #     if self._cov_matrix is None:
    #         self._cov_matrix = self.compute_cov_matrix()
    #     return self._cov_matrix

    # @cov_matrix.setter
    # def cov_matrix(self, value):
    #     self._cov_matrix = value

    def compute_cov_matrix(self):
        """
        Compute the covariance matrix of the returns of the assets.

        Args:
            returns (pd.DataFrame): The returns of the assets.
        """
        cov_matrix = self.returns.cov().values
        return cov_matrix

    # @property
    # def expected_returns(self):
    #     if self._expected_returns is None:
    #         self._expected_returns = self.compute_expected_returns()
    #     return self._expected_returns

    # @expected_returns.setter
    # def expected_returns(self, value):
    #     self._expected_returns = value

    def compute_expected_returns(self):
        """
        Compute the expected returns of the assets.

        Args:
            returns (pd.DataFrame): The returns of the assets.
        """
        expected_returns = self.returns.mean().values
        return expected_returns

    def get_allocation(self):
        """
        Get the allocation of the assets in the portfolio.
        Uses scipy's minimize function for constrained optimization. 
        The objective function is the portfolio variance.
        The constraint is that the sum of the weights is equal to 1.
        """

        from scipy.optimize import minimize

        def portfolio_variance(weights: np.array):
            """
            Function to compute the portfolio variance.
            
            Args:
                weights: np.array
                    The weights of the assets.
                    
            Returns:
                port_var: float
                    The variance of the portfolio.
            """
            cov = self.compute_cov_matrix()
            port_var = np.dot(np.dot(weights, cov), weights)
            return port_var

        def weight_constraint(weights: np.array):
            """
            Constraint function that ensures the sum of the weights is equal to 1.
            
            Args:
                weights: np.array)
                    The weights of the assets.
                
            Returns:
                constraint: float 
                    The sum of the weights minus 1.
            """
            constraint = np.sum(weights) - 1
            return constraint

        weight_bounds = [(0, 1) for _ in range(self.n_assets)] # bounds for the weights
        self.allocation = [1/self.n_assets for _ in range(self.n_assets)] # initial allocation
        constraint = {'type': 'eq', 'fun': weight_constraint}

        optimal = minimize(fun=portfolio_variance,
                        x0=self.allocation,
                        bounds=weight_bounds,
                        constraints=constraint,
                        method='SLSQP'
                        )

        self.allocation = optimal['x']

    def update_balance(self):
        """
        Update the balance of the portfolio.
        """

        self.balance = self.get_portfolio_value()
        self.portfolio_value_history.append(self.balance)

    def get_portfolio_value(self):
        """
        Get the value of the portfolio.

        Args:
            balance: float
                The initial balance of the portfolio.

        Returns:
            portfolio_value: float
                The value of the portfolio.
        """

        portfolio_value = self.balance * (1 + self.returns.iloc[-1].dot(self.allocation))
        return portfolio_value

    def plot_portfolio_value(self):
        """
        Plot the value of the portfolio over time.
        """

        plt.plot(self.portfolio_value_history)
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Time')
        plt.ylabel('Portfolio Value')
        plt.show()

