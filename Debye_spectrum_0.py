import numpy as np
from scipy.integrate import odeint

class EqnConstructor:

    def __init__(self, n, omegaD, dt):
        self.n = n
        self.omegaD = omegaD
        self.dt = dt

    @staticmethod
    def _coefficient_ap(p, n):
        return np.cos((2*p+1)*np.pi/(2*n))

    @staticmethod
    def _coefficient_bp(p, n):
        return np.sin((2 * p + 1) * np.pi / (2 * n))

    @staticmethod
    def _coefficient_sigma(n):
        return 1/(2*np.sin((3*np.pi)/(2*n)))

    def inter_force(self, xn, xl):  # Lennard-Jones force on y_l
        epsilon = 1.0  # the referencing constants of L-J potential for two Agron atoms is about 0.997kJ/mol, and
        sigma = 3.0    # 3.4 Angstroms
        temp = sigma / (xn-xl)
        return 4 * (epsilon/sigma) * (temp**13 - temp**7)


class Solver1st(EqnConstructor):  # Runge-Kutta

    # def __init__(self, n, omegaD, dt):
    #     super().__init__(n, omegaD)
    #     self.dt = dt

    # def runge_kutta(self, f, t0, y0, h):
    #
    #     k1 = h * f(t0, y0)
    #     k2 = h * f(t0 + 0.5 * h, y0 + 0.5 * k1)
    #     k3 = h * f(t0 + 0.5 * h, y0 + 0.5 * k2)
    #     k4 = h * f(t0 + h, y0 + k3)
    #     # time iteration is done in the loop outside this function
    #     return y0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def euler(self, y0):
        return y0 + y0 * self.dt

    def dsps_dt(self, p, sps, spc):
        y0 = -self._coefficient_bp(p, self.n) * self.omegaD * sps + \
             self._coefficient_ap(p, self.n) * self.omegaD * spc
        return self.euler(y0)

    def dspc_dt(self, p, spc, sps, xn, xl):
        y0 = -self._coefficient_bp(p, self.n) * self.omegaD * spc - \
             self._coefficient_ap(p, self.n) * self.omegaD * sps + \
            self.inter_force(xn, xl)
        return self.euler(y0)

    def dse_dt(self, se, xn, xl):
        y0 = -self.omegaD * se + \
             self.inter_force(xn, xl)
        return self.euler(y0)



class Solver2nd:
    pass


class FourierTransformer:
    pass


test = Solver1st(2, 10, 0.00005)
print test.dspc_dt(1, 0., 0., 1, 5)