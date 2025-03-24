import torch
from config import CONFIG
from DataRetriever import get_yfinance_data

class VanillaBlackScholes:
    def __init__(self, S0, K, T, r, sigma, type="call"):
        self.S0 = torch.tensor([S0], device=CONFIG.device, dtype=torch.float32, requires_grad=True)
        self.K = torch.tensor(K, device=CONFIG.device,dtype=torch.float32)
        self.T = torch.tensor(T, device=CONFIG.device,dtype=torch.float32)
        self.r = torch.tensor([r],device=CONFIG.device, dtype=torch.float32)
        self.sigma = sigma
        self.type = type

    def _d1_d2(self):
        d1 = (torch.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (
                    self.sigma * torch.sqrt(self.T))
        d2 = d1 - self.sigma * torch.sqrt(self.T)
        return d1, d2

    def price(self):
        d1, d2 = self._d1_d2()
        if self.type == "call":
            price = self.S0 * torch.distributions.Normal(0, 1).cdf(d1) - self.K * torch.exp(-self.r * self.T) * torch.distributions.Normal(0, 1).cdf(d2)
            return price

        elif self.type == "put":
            price = self.K * torch.exp(-self.r * self.T) * torch.distributions.Normal(0, 1).cdf(-d2) - self.S0 * torch.distributions.Normal(0, 1).cdf(-d1)
            return price




""" ------------ CALCULATE YF IV -----------------"""

def implied_vol(k_values, T_values, market_prices):
    device = CONFIG.device

    k_t = torch.tensor(k_values, device=device, dtype=torch.float32)
    T_t = torch.tensor(T_values, device=device, dtype=torch.float32)
    m_t = torch.tensor(market_prices, device=device, dtype=torch.float32)

    theta = torch.tensor([0.2] * len(k_values), device=device, requires_grad=True)

    optimizer = torch.optim.Adam([theta], lr=0.1)

    for step in range(500):
        sigma = torch.exp(theta)

        VBS = VanillaBlackScholes(
            S0=604.21, K=k_t, T=T_t, r=0.0430, sigma=sigma
        ).price()

        loss = torch.mean((VBS - m_t) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return torch.exp(theta).detach().cpu().numpy()

