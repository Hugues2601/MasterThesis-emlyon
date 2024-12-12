from HestonModel.ForwardStart import ForwardStart
from HestonModel.Vanilla import VanillaHestonPrice
import torch
import numpy as np
import matplotlib.pyplot as plt
from config import CONFIG

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

    def calculate_greek(self, greek_name: str):
        # Ensure the ForwardStart object is created once, using existing differentiable tensors
        forward_start = ForwardStart(self.S0, self.k, self.T0, self.T1, self.T2, self.r, self.kappa, self.v0, self.theta, self.sigma, self.rho)
        price = forward_start.heston_price()
        price.backward()

        if greek_name == "delta":
            return self.S0.grad.item()
        elif greek_name == "vega":
            return self.sigma.grad.item()
        elif greek_name == "theta":
            return self.T0.grad.item()
        elif greek_name == "rho":
            return self.r.grad.item()
        else:
            raise ValueError(f"Unknown greek: {greek_name}")

    # def plot_greek(self, greek_name="delta"):
    #     # Dictionary mapping greek names to their respective tensors
    #     greeks_dict = {
    #         "delta": self.S0,
    #         "vega": self.sigma,
    #         "theta": self.T0,
    #         "rho": self.r
    #     }
    #
    #     if greek_name not in greeks_dict:
    #         raise ValueError(f"Unsupported greek: {greek_name}")
    #
    #     tensor_to_differentiate = greeks_dict[greek_name]
    #
    #     k_values = torch.linspace(0.01, 2, steps=500, device=CONFIG.device)
    #     greek_values = []
    #
    #     for k in k_values:
    #         # Reset gradients before each computation
    #         if tensor_to_differentiate.grad is not None:
    #             tensor_to_differentiate.grad.zero_()
    #
    #         # Update k for the current iteration
    #         self.k = torch.tensor([k.item()], device=CONFIG.device)
    #         forward_start = ForwardStart(self.S0, self.k, self.T0, self.T1, self.T2, self.r, self.kappa, self.v0, self.theta, self.sigma, self.rho)
    #         price = forward_start.heston_price()
    #
    #         # Compute gradient
    #         price.backward()
    #
    #         grad = tensor_to_differentiate.grad
    #         if grad is None:
    #             raise RuntimeError(f"Gradient for {greek_name} is None at k={k.item()}.")
    #
    #         greek_values.append(grad.item())
    #
    #     # Convert to numpy for plotting
    #     k_values_cpu = k_values.cpu().detach().numpy()
    #     greek_values_cpu = np.array(greek_values)
    #
    #     plt.figure(figsize=(8, 6))
    #     plt.plot(k_values_cpu, greek_values_cpu, label=greek_name)
    #     plt.xlabel("Strike (k)")
    #     plt.ylabel(greek_name)
    #     plt.title(f"{greek_name} vs Strike")
    #     plt.grid(True)
    #     plt.legend()
    #     plt.show()


    # def fs_delta(S0, T0, T1, T2, r, kappa, v0, theta, sigma, rho):
    #     deltas = []
    #
    #     for kk in k_values:
    #         if S0.grad is not None:
    #             S0.grad.zero_()
    #
    #         # Compute price for the current strike
    #         price = fs_heston_price(S0, kk.unsqueeze(0), T0, T1, T2, r, kappa, v0, theta, sigma, rho)
    #
    #         price.backward()
    #         deltas.append(S0.grad.item())
    #
    #     k_values_cpu = k_values.cpu().detach().numpy()
    #     deltas_cpu = np.array(deltas)
    #
    #     # Plot Delta vs k
    #     plt.figure(figsize=(8, 6))
    #     plt.plot(k_values_cpu, deltas_cpu, label='dPrice/dS0')
    #     plt.xlabel('Strike (k)')
    #     plt.ylabel('Delta')
    #     plt.title('Delta ')
    #     plt.grid(True)
    #     plt.legend()
    #     plt.show()
    #
    #
    # def fs_gamma(S0, T0, T1, T2, r, kappa, v0, theta, sigma, rho):
    #     deltas = []
    #     gammas = []
    #
    #     for kk in k_values:
    #         # Reset gradients
    #         if S0.grad is not None:
    #             S0.grad.zero_()
    #
    #         # Compute price for the current strike
    #         price = fs_heston_price(S0, kk.unsqueeze(0), T0, T1, T2, r, kappa, v0, theta, sigma, rho)
    #
    #         # Compute the first derivative (Delta)
    #         price.backward(retain_graph=True)
    #         delta = S0.grad.item()
    #         deltas.append(delta)
    #
    #         # Reset gradients and compute the second derivative (Gamma)
    #         S0.grad.zero_()
    #         delta_tensor = torch.tensor(delta, requires_grad=True)
    #         delta_tensor.backward()
    #         gamma = S0.grad.item()
    #         gammas.append(gamma)
    #
    #     k_values_cpu = k_values.cpu().detach().numpy()
    #     gammas_cpu = np.array(gammas)
    #
    #     # Plot Gamma vs k
    #     plt.figure(figsize=(8, 6))
    #     plt.plot(k_values_cpu, gammas_cpu, label='d²Price/dS0²', color='orange')
    #     plt.xlabel('Strike (k)')
    #     plt.ylabel('Gamma')
    #     plt.title('Gamma ')
    #     plt.grid(True)
    #     plt.legend()
    #     plt.show()
    #
    #
    # def fs_vega(S0, T0, T1, T2, r, kappa, v0, theta, sigma, rho):
    #     sigmas = []
    #
    #     for kk in k_values:
    #         if sigma.grad is not None:
    #             sigma.grad.zero_()
    #
    #         # Compute price for the current strike
    #         price = fs_heston_price(S0, kk.unsqueeze(0), T0, T1, T2, r, kappa, v0, theta, sigma, rho)
    #
    #         price.backward()
    #         sigmas.append(sigma.grad.item())
    #
    #     k_values_cpu = k_values.cpu().detach().numpy()
    #     deltas_cpu = np.array(sigmas)
    #
    #     # Plot Delta vs k
    #     plt.figure(figsize=(8, 6))
    #     plt.plot(k_values_cpu, deltas_cpu, label='dPrice/dSigma')
    #     plt.xlabel('Strike (k)')
    #     plt.ylabel('Vega')
    #     plt.title('Vega ')
    #     plt.grid(True)
    #     plt.legend()
    #     plt.show()
    #
    # def fs_rho(S0, T0, T1, T2, r, kappa, v0, theta, sigma, rho):
    #     rhos = []
    #
    #     for kk in k_values:
    #         if r.grad is not None:
    #             r.grad.zero_()
    #
    #         # Compute price for the current strike
    #         price = fs_heston_price(S0, kk.unsqueeze(0), T0, T1, T2, r, kappa, v0, theta, sigma, rho)
    #
    #         price.backward()
    #         rhos.append(r.grad.item())
    #
    #     k_values_cpu = k_values.cpu().detach().numpy()
    #     deltas_cpu = np.array(rhos)
    #
    #     # Plot Delta vs k
    #     plt.figure(figsize=(8, 6))
    #     plt.plot(k_values_cpu, deltas_cpu, label='dPrice/dr')
    #     plt.xlabel('Strike (k)')
    #     plt.ylabel('Rho')
    #     plt.title('Rho ')
    #     plt.grid(True)
    #     plt.legend()
    #     plt.show()
    #
    # def fs_theta(S0, T0, T1, T2, r, kappa, v0, theta, sigma, rho):
    #     thetas = []
    #
    #     for kk in k_values:
    #         if T0.grad is not None:
    #             T0.grad.zero_()
    #
    #         # Compute price for the current strike
    #         price = fs_heston_price(S0, kk.unsqueeze(0), T0, T1, T2, r, kappa, v0, theta, sigma, rho)
    #
    #         price.backward()
    #         thetas.append(T0.grad.item())
    #
    #     k_values_cpu = k_values.cpu().detach().numpy()
    #     deltas_cpu = np.array(thetas)
    #
    #     # Plot Delta vs k
    #     plt.figure(figsize=(8, 6))
    #     plt.plot(k_values_cpu, deltas_cpu, label='dPrice/dr')
    #     plt.xlabel('Strike (k)')
    #     plt.ylabel('Theta')
    #     plt.title('Theta')
    #     plt.grid(True)
    #     plt.legend()
    #     plt.show()
    #
    # fs_delta(S0, T0, T1, T2, r, kappa, v0, theta, sigma, rho)
    # fs_vega(S0, T0, T1, T2, r, kappa, v0, theta, sigma, rho)
    # fs_rho(S0, T0, T1, T2, r, kappa, v0, theta, sigma, rho)
    # fs_gamma(S0, T0, T1, T2, r, kappa, v0, theta, sigma, rho)
    # fs_theta(S0, T0, T1, T2, r, kappa, v0, theta, sigma, rho)


