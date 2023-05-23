
from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt

class HouseholdSpecializationModelClass:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        # c. household production
        par.alpha = 0.5
        par.sigma = 1

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production
        if par.sigma == 0:
            H = optimize.minimize(HM,HF)
        elif par.sigma == 1:
            H = HM**(1-par.alpha)*HF**par.alpha
        else:
            H = ((1-par.alpha)*HM**((par.sigma-1)/par.sigma)+par.alpha*HF**((par.sigma-1)/par.sigma))**(par.sigma/(par.sigma-1))

        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
        return utility - disutility

    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt

    def solve(self, do_print=False):
        """Solve model continuously.

        Args:
            do_print (bool): Whether to print the results (default is False).

        Returns:
            An object with the optimal choices of LM, HM, LF, and HF.

        """
        # Get model parameters and solution
        par = self.par
        sol = self.sol

        # Define a function to minimize (negative of utility)
        def neg_utility(choices):
            LM, HM, LF, HF = choices
            return -self.calc_utility(LM, HM, LF, HF)

        # Set initial guesses and bounds for the optimization problem
        x0 = [2, 2, 2, 2]
        bounds = [(0, 24), (0, 24), (0, 24), (0, 24)]

        # Solve the optimization problem
        opt_result = optimize.minimize(neg_utility, x0, bounds=bounds, method='Nelder-Mead')
        opt = SimpleNamespace(
            LM=opt_result.x[0],
            HM=opt_result.x[1],
            LF=opt_result.x[2],
            HF=opt_result.x[3],
            )

        # Save the optimal choices to the solution object
        sol.LM = opt.LM
        sol.HM = opt.HM
        sol.LF = opt.LF
        sol.HF = opt.HF

        # Print the results if requested
        if do_print:
            for k, v in opt.__dict__.items():
                print(f"{k} = {v:6.4f}")

        return opt

    def solve_wF_vec(self, discrete=False):
        """ solve model for vector of female wages """
        par = self.par
        sol = self.sol

        HF_HM_ratios = np.zeros(par.wF_vec.size)

        for i, wF in enumerate(par.wF_vec):

            # update female wage
            par.wF = wF

            # solve the model
            if discrete:
                opt = self.solve_discrete()
            else:
                opt = self.solve()

            # store the results
            sol.LM_vec[i] = opt.LM
            sol.HM_vec[i] = opt.HM
            sol.LF_vec[i] = opt.LF
            sol.HF_vec[i] = opt.HF

            # calculate HF/HM ratio
            sol.HM = opt.HM
            sol.HF = opt.HF
            HF_HM_ratios[i] = sol.HF / sol.HM

        # store HF/HM ratios in solution object
        sol.HF_HM_ratios = HF_HM_ratios

        # run regression if not discrete
        if not discrete:
            self.run_regression()

    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]
    
    def estimate(self, do_print=False):
        par = self.par
        sol = self.sol

        # Initial guess and bounds for the optimization
        x0 = [0.5, 0.1]
        bounds = ((0, 1), (0, 1))

        # Define the target function for optimization
        def target(x):
            par.alpha, par.sigma = x
            self.solve_wF_vec()
            self.run_regression()
            residual = (sol.beta0 - par.beta0_target) ** 2 + (sol.beta1 - par.beta1_target) ** 2
            return residual

        # Perform the optimization
        solution = optimize.minimize(target, x0, method='Nelder-Mead', bounds=bounds)
        sol.alpha, sol.sigma = solution.x

        if do_print:
            # Print the optimization results
            print("Optimization results:")
            print("=====================")
            print(f"Alpha: {sol.alpha:.2f}")
            print(f"Sigma: {sol.sigma:.2f}")
            print(f"Residual value: {solution.fun:.2f}")
            print(f"Success: {solution.success}")
            print(f"Message: {solution.message}")

    def calc_utility5(self, LM, HM, LF, HF):
        """ calculate utility with an inequality aversion component """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM * LM + par.wF * LF

        # b. home production
        if par.sigma == 0:
            H = optimize.minimize(HM, HF)
        elif par.sigma == 1:
            H = HM**(1 - par.alpha) * HF**par.alpha
        else:
            H = ((1 - par.alpha) * HM**((par.sigma - 1) / par.sigma) + par.alpha * HF**((par.sigma - 1) / par.sigma))**(par.sigma / (par.sigma - 1))

        # c. total consumption utility
        Q = C**par.omega * H**(1 - par.omega)
        utility = np.fmax(Q, 1e-8)**(1 - par.rho) / (1 - par.rho)

        # d. disutility of work
        epsilon_ = 1 + 1 / par.epsilon
        TM = LM + HM
        TF = LF + HF
        disutility = par.nu * (TM**epsilon_ / epsilon_ + TF**epsilon_ / epsilon_)

        # e. inequality aversion component
        inequality_aversion = par.lambda_ineq * (HF - HM)**2  # Adjust the parameter lambda_ineq as desired

        return utility - disutility - inequality_aversion
    
    def estimate5(self, do_print=False):
        par = self.par
        sol = self.sol

        # Initial guess and bounds for the optimization
        x0 = [0.5, 0.1]
        bounds = ((0.5, 0.5), (0, 1))

        # Define the target function for optimization
        def target(x):
            par.alpha, par.sigma = x
            self.solve_wF_vec5()
            self.run_regression()
            residual = (sol.beta0 - par.beta0_target) ** 2 + (sol.beta1 - par.beta1_target) ** 2
            return residual

        # Perform the optimization
        solution = optimize.minimize(target, x0, method='Nelder-Mead', bounds=bounds)
        sol.alpha, sol.sigma = solution.x

        if do_print:
            # Print the optimization results
            print("Optimization results:")
            print("=====================")
            print(f"Alpha: {sol.alpha:.2f}")
            print(f"Sigma: {sol.sigma:.2f}")
            print(f"Residual value: {solution.fun:.2f}")
            print(f"Success: {solution.success}")
            print(f"Message: {solution.message}")

    def solve_wF_vec5(self, discrete=False):
        """ solve model for vector of female wages """
        par = self.par
        sol = self.sol

        HF_HM_ratios = np.zeros(par.wF_vec.size)

        for i, wF in enumerate(par.wF_vec):

            # update female wage
            par.wF = wF

            # solve the model
            if discrete:
                opt = self.solve_discrete()
            else:
                opt = self.solve5()

            # store the results
            sol.LM_vec[i] = opt.LM
            sol.HM_vec[i] = opt.HM
            sol.LF_vec[i] = opt.LF
            sol.HF_vec[i] = opt.HF

            # calculate HF/HM ratio
            sol.HM = opt.HM
            sol.HF = opt.HF
            HF_HM_ratios[i] = sol.HF / sol.HM

        # store HF/HM ratios in solution object
        sol.HF_HM_ratios = HF_HM_ratios

        # run regression if not discrete
        if not discrete:
            self.run_regression()

    def solve5(self, do_print=False):
        """Solve model continuously.

        Args:
            do_print (bool): Whether to print the results (default is False).

        Returns:
            An object with the optimal choices of LM, HM, LF, and HF.

        """
        # Get model parameters and solution
        par = self.par
        sol = self.sol

        # Define a function to minimize (negative of utility)
        def neg_utility(choices):
            LM, HM, LF, HF = choices
            return -self.calc_utility5(LM, HM, LF, HF)

        # Set initial guesses and bounds for the optimization problem
        x0 = [2, 2, 2, 2]
        bounds = [(0, 24), (0, 24), (0, 24), (0, 24)]

        # Solve the optimization problem
        opt_result = optimize.minimize(neg_utility, x0, bounds=bounds, method='Nelder-Mead')
        opt = SimpleNamespace(
            LM=opt_result.x[0],
            HM=opt_result.x[1],
            LF=opt_result.x[2],
            HF=opt_result.x[3],
            )

        # Save the optimal choices to the solution object
        sol.LM = opt.LM
        sol.HM = opt.HM
        sol.LF = opt.LF
        sol.HF = opt.HF

        # Print the results if requested
        if do_print:
            for k, v in opt.__dict__.items():
                print(f"{k} = {v:6.4f}")

        return opt