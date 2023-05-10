import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interactive, fixed
from types import SimpleNamespace
import scipy.optimize as optimize



class Romer:
    def __init__(self):
        par = self.par = SimpleNamespace()

        # Parameters
        par.alpha = 0.3
        par.beta = 0.2
        par.kappa = 1 - par.alpha - par.beta
        par.rho = 0.05
        par.phi = 0.5
        par.g = 0.02
        par.lambda_ = 0.5
        par.s = 0.2
        par.delta = 0.05
        par.n = 0.02
        par.s_R = 0.2
        par.X = 1.0 # land as a fixed resource
        par.L = 1  # land is fixed

    def steady_state_equations(x, par):
        A, K, L, LA = x
        eq1 = A - ((1+par.rho)/(1+par.n+par.rho))**(1/(1-par.phi))
        eq2 = K - (par.s/(par.n+par.delta))**(1/(1-par.alpha)) * A * (par.s_R+1)
        eq3 = L - K**(1-par.alpha)/(A**(1-par.alpha))
        eq4 = LA - par.s_R*L
        return [eq1, eq2, eq3, eq4]

    def steady_state_values(self):
        par = self.par
        x0 = [1.0, 1.0, 1.0, 1.0] # initial guess for A, K, L, and LA
        sol = optimize.root(Romer.steady_state_equations, x0, args=(par,), method='hybr')
        A, K, L, LA = sol.x
        Y = A*K**par.alpha*L**(1-par.alpha)
        Y_per_L = Y/L
        return Y_per_L, Y, L, K, A, LA

class Solow:
    def __init__(self):
        par = self.par = SimpleNamespace()

        # Parameters
        par.alpha = 0.3
        par.beta = 0.2
        par.kappa = 1 - par.alpha - par.beta
        par.rho = 0.05
        par.phi = 0.5
        par.g = 0.02
        par.lambda_ = 0.5
        par.s = 0.2
        par.delta = 0.05
        par.n = 0.02
        par.s_R = 0.2
        par.X = 1.0 # land as a fixed resource
        par.L = 1  # land is fixed
    
    def steady_state_equations(x, par):
        K, A = x
        if K==0:
            eq1 = 0  # set output to 0 when capital is 0
        else:
            eq1 = K**par.alpha*(A*par.L)**par.beta*par.X**par.kappa - par.s*K - par.delta*K
        eq2 = A - (1+par.g)*A
        return [eq1, eq2]

    def steady_state_values(self):
        par = self.par
        x0 = [4.0, 4.0] # initial guess for K and A
        sol = optimize.root(Solow.steady_state_equations, x0, args=(par,), method='hybr')
        K_s, A_s = sol.x
        Y_s = K_s**par.alpha*(A_s*par.L)**par.beta*par.X**par.kappa
        Y_per_L_s = Y_s/par.L
        return Y_per_L_s, Y_s, K_s, A_s

class SemiEndogenousRomer:
    def __init__(self):
        par = self.par = SimpleNamespace()

        # Parameters
        par.alpha = 0.3
        par.beta = 0.2
        par.kappa = 1 - par.alpha - par.beta
        par.rho = 0.05
        par.phi = 0.5
        par.g = 0.02
        par.lambda_ = 0.5
        par.s = 0.2
        par.delta = 0.05
        par.n = 0.02
        par.s_R = 0.2
        par.X = 1.0 # land as a fixed resource
        par.L = 1  # land is fixed

    def steady_state_equations(self, x):
        par = self.par
        A, K, L, LA = x
        eq1 = A - ((1+par.rho)/(1+par.n+par.rho))**(1/(1-par.phi))
        eq2 = K - (par.s/(par.n+par.delta))**(1/(1-par.alpha)) * A**(par.alpha/(1-par.alpha)) * L**(par.beta/(1-par.alpha)) * (par.s_R*par.X+L)**(par.kappa/(1-par.alpha))
        eq3 = L - par.X - LA
        eq4 = par.s_R * L - LA
        return [eq1, eq2, eq3, eq4]

    def steady_state_values(self):
        par = self.par
        x0 = [1.0, 1.0, 1.0, 1.0] # initial guess for A, K, L, and LA
        sol = optimize.root(self.steady_state_equations, x0, method='hybr')
        A, K, L, LA = sol.x
        Y = K**par.alpha*(A*L)**par.beta*par.X**par.kappa
        Y_per_L = Y/L
        return Y_per_L, Y, L, K, A, LA