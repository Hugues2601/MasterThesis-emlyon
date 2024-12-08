from HestonModel.ForwardStart import heston_price
import torch
from config import CONFIG



Kappa = torch.tensor(2, device=CONFIG.device)
v0 = torch.tensor(0.04, device=CONFIG.device)
Theta = torch.tensor(0.04, device=CONFIG.device)
sigma = torch.tensor(0.3, device=CONFIG.device)
rho = torch.tensor(-0.7, device=CONFIG.device)
k=torch.tensor([100], device=CONFIG.device)
S0 = torch.tensor(100, device=CONFIG.device)  # Initial stock price
r = torch.tensor(0.05, device=CONFIG.device)
T = torch.tensor([2], device=CONFIG.device)
T0 = torch.tensor([1], device=CONFIG.device)# Ri
t0 = torch.tensor([0], device=CONFIG.device)

print(heston_price(S0, k, t0, T0, T, r, Kappa, v0, Theta, sigma, rho ))