from HestonModel.ForwardStart import fs_heston_price
from HestonModel.Vanilla import heston_price
import torch
import numpy as np
import matplotlib.pyplot as plt
from config import CONFIG

parameters = {'kappa': 4.142323253439034, 'v0': 0.012856513424731748, 'theta': 0.007792565017067254, 'sigma': 0.07402665592690841, 'rho': -0.8894383787135084}
S0 = torch.tensor(100.0, device=CONFIG.device, requires_grad=True)
k = torch.tensor([1], device=CONFIG.device)
# K= torch.tensor([100], device=CONFIG.device)
T0 = torch.tensor([0.0], device=CONFIG.device, requires_grad=True)
T1 = torch.tensor([1], device=CONFIG.device)
T2 = torch.tensor([2], device=CONFIG.device)
T = torch.tensor([1], device=CONFIG.device)
r = torch.tensor(0.0, device=CONFIG.device, requires_grad=True)
kappa = torch.tensor(2, device=CONFIG.device)
v0 = torch.tensor(0.04, device=CONFIG.device)
theta = torch.tensor(0.04, device=CONFIG.device)
sigma = torch.tensor(0.2, device=CONFIG.device, requires_grad=True)
rho = torch.tensor(-0.6, device=CONFIG.device)

price = fs_heston_price(S0, k, T0, T1, T2, r, kappa, v0, theta, sigma, rho)
# priceh = heston_price(S0, K, T, r, kappa, v0, theta, sigma, rho)
print(f"prix de l'option FS : {price} | prix Heston : {None}")
k_values = torch.linspace(0.01, 2, steps=500, device=CONFIG.device)


class GreeksFS:
    def __init__(self, S0, k, T0, T1, T2, r, kappa, v0, theta, sigma, rho):
        self.S0 = torch.tensor(S0, device=CONFIG.device, requires_grad=True)
        self.k = torch.tensor([k], device=CONFIG.device)
        self.T0 = torch.tensor([T0], device=CONFIG.device, requires_grad=True)
        self.T1 = torch.tensor([T1], device=CONFIG.device)
        self.T2 = torch.tensor([T2], device=CONFIG.device)
        self.r = torch.tensor(r, device=CONFIG.device, requires_grad=True)
        self.kappa = torch.tensor(kappa, device=CONFIG.device)
        self.v0 = torch.tensor(v0, device=CONFIG.device)
        self.theta = torch.tensor(theta, device=CONFIG.device)
        self.sigma = torch.tensor(sigma, device=CONFIG.device, requires_grad=True)
        self.rho = torch.tensor(rho, device=CONFIG.device)

    def calculate_greek(self, greek_name = "delta"):
        price = heston_price(self.S0, self.T0, self.T1, self.T2, self.r, self.kappa, self.v0, self.theta, self.rho)
        price.backward()
        greek = []
        if greek_name == "delta":
            delta = S0.grad.item()
            greek.append(delta)

        elif greek_name == "rho":
            rho = r.grad.item()
            greek.append(rho)

        elif greek_name == "vega":
            vega = sigma.grad.item()
            greek.append(vega)

        elif greek_name == "theta":
            theta = T0.grad.item()
            greek.append(theta)

        elif greek_name == "gamma":
            delta = torch.autograd.grad(price, self.S0, create_graph=True)[0]
            gamma = torch.autograd.grad(delta, self.S0)[0]
            greek.append(gamma)

        self.S0.grad = None
        self.T0.grad = None
        self.r.grad = None
        self.sigma.grad = None

        return f"{greek_name}: {greek[0]}"

    def plot_greek(self, greek_name = "delta"):
        k_values = torch.linspace(0.01, 2, steps=500, device=CONFIG.device)

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
        plt.plot(k_values_cpu, deltas_cpu, label='dPrice/dS0')
        plt.xlabel('Strike (k)')
        plt.ylabel('Delta')
        plt.title('Delta ')
        plt.grid(True)
        plt.legend()
        plt.show()


    def fs_gamma(S0, T0, T1, T2, r, kappa, v0, theta, sigma, rho):
        deltas = []
        gammas = []

        for kk in k_values:
            # Reset gradients
            if S0.grad is not None:
                S0.grad.zero_()

            # Compute price for the current strike
            price = fs_heston_price(S0, kk.unsqueeze(0), T0, T1, T2, r, kappa, v0, theta, sigma, rho)

            # Compute the first derivative (Delta)
            price.backward(retain_graph=True)
            delta = S0.grad.item()
            deltas.append(delta)

            # Reset gradients and compute the second derivative (Gamma)
            S0.grad.zero_()
            delta_tensor = torch.tensor(delta, requires_grad=True)
            delta_tensor.backward()
            gamma = S0.grad.item()
            gammas.append(gamma)

        k_values_cpu = k_values.cpu().detach().numpy()
        gammas_cpu = np.array(gammas)

        # Plot Gamma vs k
        plt.figure(figsize=(8, 6))
        plt.plot(k_values_cpu, gammas_cpu, label='d²Price/dS0²', color='orange')
        plt.xlabel('Strike (k)')
        plt.ylabel('Gamma')
        plt.title('Gamma ')
        plt.grid(True)
        plt.legend()
        plt.show()


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
        plt.plot(k_values_cpu, deltas_cpu, label='dPrice/dSigma')
        plt.xlabel('Strike (k)')
        plt.ylabel('Vega')
        plt.title('Vega ')
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
        plt.plot(k_values_cpu, deltas_cpu, label='dPrice/dr')
        plt.xlabel('Strike (k)')
        plt.ylabel('Rho')
        plt.title('Rho ')
        plt.grid(True)
        plt.legend()
        plt.show()

    def fs_theta(S0, T0, T1, T2, r, kappa, v0, theta, sigma, rho):
        thetas = []

        for kk in k_values:
            if T0.grad is not None:
                T0.grad.zero_()

            # Compute price for the current strike
            price = fs_heston_price(S0, kk.unsqueeze(0), T0, T1, T2, r, kappa, v0, theta, sigma, rho)

            price.backward()
            thetas.append(T0.grad.item())

        k_values_cpu = k_values.cpu().detach().numpy()
        deltas_cpu = np.array(thetas)

        # Plot Delta vs k
        plt.figure(figsize=(8, 6))
        plt.plot(k_values_cpu, deltas_cpu, label='dPrice/dr')
        plt.xlabel('Strike (k)')
        plt.ylabel('Theta')
        plt.title('Theta')
        plt.grid(True)
        plt.legend()
        plt.show()

    fs_delta(S0, T0, T1, T2, r, kappa, v0, theta, sigma, rho)
    fs_vega(S0, T0, T1, T2, r, kappa, v0, theta, sigma, rho)
    fs_rho(S0, T0, T1, T2, r, kappa, v0, theta, sigma, rho)
    fs_gamma(S0, T0, T1, T2, r, kappa, v0, theta, sigma, rho)
    fs_theta(S0, T0, T1, T2, r, kappa, v0, theta, sigma, rho)


