import torch


def heston_cf(u, t, tau, hestonParams):

    kappa, theta, sigma, rho, v0 = hestonParams

    d = torch.sqrt((kappa - rho * sigma * u) * (kappa - rho * sigma * u) + u * (1. - u) * sigma * sigma)

    Gamma = (kappa - rho * sigma * u - d) / (kappa - rho * sigma * u + d)

    A = ((kappa * theta) / (sigma * sigma)) * (
                (kappa - rho * sigma * u - d) * tau - 2. * torch.log((1. - Gamma * torch.exp(-d * tau)) / (1. - Gamma)))

    B = (1. / (sigma * sigma)) * (kappa - rho * sigma * u - d) * (
                (1. - torch.exp(-d * tau)) / (1. - Gamma * torch.exp(-d * tau)))

    BBb = ((sigma * sigma) / (4. * kappa)) * (1. - torch.exp(-kappa * t))

    cf_val = torch.exp(A
                       + (B / (1. - 2. * BBb * B)) * (v0 * torch.exp(-kappa * t))
                       - ((2. * kappa * theta) / (sigma * sigma)) * torch.log(1. - 2. * BBb * B))
    return cf_val


def heston_price(S0, K, t, tau, hestonParams, precision=20.0, N=2000):

    kappa, theta, sigma, rho, v0 = hestonParams
    alpha = 0.5
    k = torch.log(K)

    pp = heston_cf(torch.tensor(1.0, dtype=torch.complex64), t, tau, hestonParams).real

    w = torch.linspace(-precision, precision, N, dtype=torch.float32)
    dw = w[1] - w[0]

    u = alpha + 1j * w
    cf_val = heston_cf(u, t, tau, hestonParams)

    denom = (u * (1. - alpha - 1j * w))
    intPhi_val = torch.exp(-k * u) * cf_val / denom
    intPhi_val_real = intPhi_val.real

    integral = torch.trapz(intPhi_val_real, w)

    price = S0*(pp - K / (2. * torch.pi) * integral)

    return price.real


kappa = torch.tensor(2.1, dtype=torch.float64)
theta = torch.tensor(0.03, dtype=torch.float64)
sigma = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
rho   = torch.tensor(-0.2, dtype=torch.float64)
v0    = torch.tensor(0.05, dtype=torch.float64)
S0 = torch.tensor(100, dtype=torch.float64, requires_grad=True)

hestonParams = (kappa, theta, sigma, rho, v0)

k    = torch.tensor(1, dtype=torch.float64)
t    = torch.tensor(1, dtype=torch.float64, requires_grad=True)
tau  = torch.tensor(2, dtype=torch.float64, requires_grad=True)
precision = 20.0

price = heston_price(S0, k, t, tau, hestonParams, precision=precision)

K_list = torch.linspace(0.05, 2, 100, dtype=torch.float64)  # par exemple 20 points entre 0.05 et 1.5

deltas = []
vegas = []
thetas = []


for K_ in K_list:
    # On crée un nouveau S0 avec gradient activé
    S0_ = torch.tensor(100, dtype=torch.float64, requires_grad=True)

    # Calcul du prix
    price = heston_price(S0_, K_, t, tau, hestonParams, precision=precision)

    # On remet à zéro les gradients si nécessaire (pas obligé si S0_ est créé chaque fois)
    if S0_.grad is not None:
        S0_.grad.zero_()
    if sigma.grad is not None:
        sigma.grad.zero_()
    if t.grad is not None:
        tau.grad.zero_()

    # Backprop pour calculer dPrice/dS0
    price.backward()

    # Récupération du delta
    delta = S0_.grad.item()
    vega = sigma.grad.item()  # dPrice/dsigma
    theta_ = t.grad.item()  # dPrice/dtau

    deltas.append(delta)
    vegas.append(vega)
    thetas.append(theta_)

import matplotlib.pyplot as plt

# Plot du Delta
plt.figure(figsize=(10,6))
plt.plot(K_list, deltas, 'o', label="Delta")
plt.xlabel("Strike (K)")
plt.ylabel("Delta")
plt.title("Delta en fonction du Strike")
plt.grid(True)
plt.legend()
plt.show()

# Plot du Vega
plt.figure(figsize=(10,6))
plt.plot(K_list, vegas, 'o', color='red', label="Vega")
plt.xlabel("Strike (K)")
plt.ylabel("Vega")
plt.title("Vega en fonction du Strike")
plt.grid(True)
plt.legend()
plt.show()

# Plot du Theta
plt.figure(figsize=(10,6))
plt.plot(K_list, thetas, 'o', color='green', label="Theta")
plt.xlabel("Strike (K)")
plt.ylabel("Theta")
plt.title("Theta en fonction du Strike")
plt.grid(True)
plt.legend()
plt.show()