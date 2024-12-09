from HestonModel.Vanilla import heston_price
import torch
import numpy as np
import matplotlib.pyplot as plt
from config import CONFIG



S0 = torch.tensor(100.0, device=CONFIG.device, requires_grad=True)
k = torch.tensor([0.3], device=CONFIG.device)
K= torch.tensor([100], device=CONFIG.device)
T0 = torch.tensor([0], device=CONFIG.device)
T1 = torch.tensor([1], device=CONFIG.device)
T2 = torch.tensor([3], device=CONFIG.device)
T = torch.tensor([2], device=CONFIG.device)
r = torch.tensor(0.04, device=CONFIG.device, requires_grad=True)
kappa = torch.tensor(4.14, device=CONFIG.device)
v0 = torch.tensor(0.01, device=CONFIG.device)
theta = torch.tensor(0.01, device=CONFIG.device)
sigma = torch.tensor(0.07, device=CONFIG.device, requires_grad=True)
rho = torch.tensor(-0.89, device=CONFIG.device)

k_values = torch.linspace(1, 200, steps=500, device=CONFIG.device)
def fs_rho(S0, T, r, kappa, v0, theta, sigma, rho):
    rhos = []

    for kk in k_values:
        if r.grad is not None:
            r.grad.zero_()

        # Compute price for the current strike
        price = heston_price(S0, kk.unsqueeze(0), T, r, kappa, v0, theta, sigma, rho)

        price.backward()
        rhos.append(r.grad.item())

    k_values_cpu = k_values.cpu().detach().numpy()
    deltas_cpu = np.array(rhos)

    # Plot Delta vs k
    plt.figure(figsize=(8, 6))
    plt.plot(k_values_cpu, deltas_cpu, label='dPrice/dr')
    plt.xlabel('Strike (k)')
    plt.ylabel('Rho')
    plt.title('Rho ')
    plt.grid(True)
    plt.legend()
    plt.show()

fs_rho(S0, T, r, kappa, v0, theta, sigma, rho)