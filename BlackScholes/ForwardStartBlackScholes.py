import torch
import math
from config import CONFIG

class FSBlackScholes:
    def __init__(self, S0, k, T1, T2, r, sigma):
        self.S0 = torch.tensor([S0], device=CONFIG.device, dtype=torch.float32, requires_grad=True)
        self.k = torch.tensor(k, device=CONFIG.device,dtype=torch.float32)
        self.T1 = torch.tensor([T1],device=CONFIG.device, dtype=torch.float32)
        self.T2 = torch.tensor(T2,device=CONFIG.device, dtype=torch.float32)
        self.r = torch.tensor([r],device=CONFIG.device, dtype=torch.float32)
        self.sigma = sigma

    def _d1_d2(self):
        tau = self.T2 - self.T1
        d1 = (torch.log(1 / self.k) + (self.r + 0.5 * self.sigma ** 2) * tau) / (
                    self.sigma * torch.sqrt(tau))
        d2 = d1 - self.sigma * torch.sqrt(tau)
        return d1, d2

    def price(self):
        tau = self.T2 - self.T1
        d1, d2 = self._d1_d2()
        price = self.S0 * (torch.distributions.Normal(0, 1).cdf(d1) - self.k * torch.exp(-self.r * tau) * torch.distributions.Normal(0, 1).cdf(d2))
        return price
