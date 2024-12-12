from abc import ABC, abstractmethod
import torch
from config import CONFIG

class HestonModel(ABC):
    def __init__(self, S0, K, T, r, kappa, v0, theta, sigma, rho):
        self.S0 = torch.tensor(S0, device=CONFIG.device, requires_grad=True)
        self.K = torch.tensor([K], device=CONFIG.device)
        self.T = torch.tensor([T], device=CONFIG.device, requires_grad=True)
        self.r = torch.tensor(r, device=CONFIG.device, requires_grad=True)
        self.kappa = torch.tensor(kappa, device=CONFIG.device)
        self.v0 = torch.tensor(v0, device=CONFIG.device)
        self.theta = torch.tensor(theta, device=CONFIG.device)
        self.sigma = torch.tensor(sigma, device=CONFIG.device, requires_grad=True)
        self.rho = torch.tensor(rho, device=CONFIG.device)

    @abstractmethod
    def _heston_cf(self, phi):
        pass

    @abstractmethod
    def heston_price(self):
        pass