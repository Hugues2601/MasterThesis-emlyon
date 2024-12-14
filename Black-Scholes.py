import torch
import math

class BlackScholes:
    def __init__(self, S0, K, T, r, sigma):
        self.S0 = torch.tensor([S0], dtype=torch.float64, requires_grad=True)
        self.K = torch.tensor([K], dtype=torch.float64)
        self.T = torch.tensor([T], dtype=torch.float64)
        self.r = torch.tensor([r], dtype=torch.float64)
        self.sigma = torch.tensor([sigma], dtype=torch.float64)

    def _d1_d2(self):
        d1 = (torch.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (
                    self.sigma * torch.sqrt(self.T))
        d2 = d1 - self.sigma * torch.sqrt(self.T)
        return d1, d2

    def price(self):
        d1, d2 = self._d1_d2()
        price = (
                self.S0 * torch.distributions.Normal(0, 1).cdf(d1)
                - self.K * torch.exp(-self.r * self.T) * torch.distributions.Normal(0, 1).cdf(d2)
        )
        return price
