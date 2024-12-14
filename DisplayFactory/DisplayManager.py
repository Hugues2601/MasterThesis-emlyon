from abc import abstractmethod
from abc import ABC

class DisplayManager(ABC):
    def __init__(self, S0, k, T0, T1, T2, r, kappa, v0, theta, sigma, rho):
        self.S0 = S0
        self.k = k
        self.r = r
        self.T0 = T0
        self.T1 = T1
        self.T2 = T2
        self.kappa = kappa
        self.v0 = v0
        self.theta = theta
        self.sigma = sigma
        self.rho = rho

    @abstractmethod
    def _dataProcessing(self):
        pass

    @abstractmethod
    def display(self):
        pass