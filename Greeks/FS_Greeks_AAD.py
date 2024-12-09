from HestonModel.ForwardStart import fs_heston_price
import torch
import numpy as np
import matplotlib.pyplot as plt
from config import CONFIG

S0 = torch.tensor(100.0, device=CONFIG.device, requires_grad=True)
k = torch.tensor([1], device=CONFIG.device)
T0 = torch.tensor([0], device=CONFIG.device)
T1 = torch.tensor([1], device=CONFIG.device)
T2 = torch.tensor([2], device=CONFIG.device)
r = torch.tensor(0.0, device=CONFIG.device, requires_grad=True)
kappa = torch.tensor(2, device=CONFIG.device)
v0 = torch.tensor(0.04, device=CONFIG.device)
theta = torch.tensor(0.04, device=CONFIG.device)
sigma = torch.tensor(0.2, device=CONFIG.device, requires_grad=True)
rho = torch.tensor(-0.7, device=CONFIG.device)

price = fs_heston_price(S0, k, T0, T1, T2, r, kappa, v0, theta, sigma, rho)
k_values = torch.linspace(0.05, 2, steps=100, device=CONFIG.device)


def fs_delta(S0, T0, T1, T2, r, kappa, v0, theta, sigma, rho):
    deltas = []

    for kk in k_values:
        if S0.grad is not None:
            S0.grad.zero_()

        # Compute price for the current strike
        price = fs_heston_price(S0, kk.unsqueeze(0), T0, T1, T2, r, kappa, v0, theta, sigma, rho)

        price.backward()
        deltas.append(S0.grad.item())

    k_values_cpu = k_values.cpu().detach().numpy()
    deltas_cpu = np.array(deltas)

    # Plot Delta vs k
    plt.figure(figsize=(8, 6))
    plt.plot(k_values_cpu, deltas_cpu, label='Delta')
    plt.xlabel('Strike (k)')
    plt.ylabel('Delta')
    plt.title('Delta as a Function of Strike')
    plt.grid(True)
    plt.legend()
    plt.show()


# def fs_gamma(S0, k_values, T0, T1, T2, r, kappa, v0, theta, sigma, rho):
#     gammas = []
#
#     for kk in k_values:
#         # Remise à zéro du gradient sur S0
#         if S0.grad is not None:
#             S0.grad.zero_()
#
#         # Calcul du prix pour la valeur de k actuelle
#         price = fs_heston_price(S0, kk.unsqueeze(0), T0, T1, T2, r, kappa, v0, theta, sigma, rho)
#
#         # Première dérivée (Delta) avec create_graph=True
#         delta = torch.autograd.grad(price, S0, create_graph=True)[0]
#
#         # Seconde dérivée (Gamma)
#         # Ici, pas besoin de create_graph, sauf si on veut des dérivées d'ordre supérieur
#         gamma = torch.autograd.grad(delta, S0, retain_graph=False)[0]
#
#         gammas.append(gamma.item())
#
#     k_values_cpu = k_values.cpu().detach().numpy()
#     gammas_cpu = np.array(gammas)
#
#     # Tracé du gamma en fonction de k
#     plt.figure(figsize=(8, 6))
#     plt.plot(k_values_cpu, gammas_cpu, label='Gamma')
#     plt.xlabel('Strike (k)')
#     plt.ylabel('Gamma')
#     plt.title('Gamma as a Function of Strike')
#     plt.grid(True)
#     plt.legend()
#     plt.show()



def fs_vega(S0, T0, T1, T2, r, kappa, v0, theta, sigma, rho):
    sigmas = []

    for kk in k_values:
        if sigma.grad is not None:
            sigma.grad.zero_()

        # Compute price for the current strike
        price = fs_heston_price(S0, kk.unsqueeze(0), T0, T1, T2, r, kappa, v0, theta, sigma, rho)

        price.backward()
        sigmas.append(sigma.grad.item())

    k_values_cpu = k_values.cpu().detach().numpy()
    deltas_cpu = np.array(sigmas)

    # Plot Delta vs k
    plt.figure(figsize=(8, 6))
    plt.plot(k_values_cpu, deltas_cpu, label='Delta')
    plt.xlabel('Strike (k)')
    plt.ylabel('Vega')
    plt.title('Vega as a Function of Strike')
    plt.grid(True)
    plt.legend()
    plt.show()

def fs_rho(S0, T0, T1, T2, r, kappa, v0, theta, sigma, rho):
    rhos = []

    for kk in k_values:
        if r.grad is not None:
            r.grad.zero_()

        # Compute price for the current strike
        price = fs_heston_price(S0, kk.unsqueeze(0), T0, T1, T2, r, kappa, v0, theta, sigma, rho)

        price.backward()
        rhos.append(r.grad.item())

    k_values_cpu = k_values.cpu().detach().numpy()
    deltas_cpu = np.array(rhos)

    # Plot Delta vs k
    plt.figure(figsize=(8, 6))
    plt.plot(k_values_cpu, deltas_cpu, label='Delta')
    plt.xlabel('Strike (k)')
    plt.ylabel('Rho')
    plt.title('Rho as a Function of Strike')
    plt.grid(True)
    plt.legend()
    plt.show()

fs_vega(S0, T0, T1, T2, r, kappa, v0, theta, sigma, rho)
fs_rho(S0, T0, T1, T2, r, kappa, v0, theta, sigma, rho)


