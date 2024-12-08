import torch
from config import *

""" ----------------------- Characteristic Function --------------------"""

def heston_cf(phi, S0, T, r, kappa, v0, theta, sigma, rho):
    # Ensure that phi is a torch tensor on the GPU
    if not isinstance(phi, torch.Tensor):
        phi = torch.tensor(phi, dtype=torch.complex128, device=CONFIG.device)
    else:
        phi = phi.to(CONFIG.device).type(torch.complex128)


    S0 = S0.to(CONFIG.device).type(torch.float64)
    T = T.to(CONFIG.device).type(torch.float64)
    r = r.to(CONFIG.device).type(torch.float64)

    a = -0.5 * phi ** 2 - 0.5 * 1j * phi
    b = kappa - rho * sigma * 1j * phi

    g = ((b - torch.sqrt(b ** 2 - 2 * sigma ** 2 * a)) / sigma ** 2) / ((b + torch.sqrt(b ** 2 - 2 * sigma ** 2 * a)) / sigma ** 2)

    C = kappa * (((b - torch.sqrt(b ** 2 - 2 * sigma ** 2 * a)) / sigma ** 2) * T - 2 / sigma ** 2 * torch.log(
        (1 - g * torch.exp(-torch.sqrt(b ** 2 - 2 * sigma ** 2 * a) * T)) / (1 - g)))

    D = ((b - torch.sqrt(b ** 2 - 2 * sigma ** 2 * a)) / sigma ** 2) * (
                1 - torch.exp(-torch.sqrt(b ** 2 - 2 * sigma ** 2 * a) * T)) / (
                    1 - g * torch.exp(-torch.sqrt(b ** 2 - 2 * sigma ** 2 * a) * T))

    cf = torch.exp(C * theta + D * v0 + 1j * phi * torch.log(S0 * torch.exp(r * T)))

    return cf


""" ---------------------- Pricing Formula ----------------------"""

def heston_price(S0, K, T, r, kappa, v0, theta, sigma, rho):

    # Vérification de la taille de K et T
    assert K.dim() == 1, "K doit être un tenseur 1D"
    assert T.dim() == 1, "T doit être un tenseur 1D"
    assert len(K) == len(T), "K et T doivent avoir la même taille"


    umax = 50
    n = 100
    if n % 2 == 0:
        n += 1

    phi_values = torch.linspace(1e-5, umax, n, device=K.device)
    du = (umax - 1e-5) / (n - 1)

    phi_values = phi_values.unsqueeze(1).repeat(1, len(K))

    factor1 = torch.exp(-1j * phi_values * torch.log(K))
    denominator = 1j * phi_values


    cf1 = heston_cf(phi_values - 1j, S0, T, r, kappa, v0, theta, sigma, rho) / heston_cf(-1j, S0, T, r, kappa, v0, theta, sigma, rho)
    temp1 = factor1 * cf1 / denominator
    integrand_P1_values = 1 / torch.pi * torch.real(temp1)


    cf2 = heston_cf(phi_values, S0, T, r, kappa, v0, theta, sigma, rho)
    temp2 = factor1 * cf2 / denominator
    integrand_P2_values = 1 / torch.pi * torch.real(temp2)

    weights = torch.ones(n, device=K.device)
    weights[1:-1:2] = 4
    weights[2:-2:2] = 2
    weights *= du / 3
    weights = weights.unsqueeze(1).repeat(1, len(K))

    integral_P1 = torch.sum(weights * integrand_P1_values, dim=0)
    integral_P2 = torch.sum(weights * integrand_P2_values, dim=0)

    P1 = torch.tensor(0.5, device=K.device) + integral_P1
    P2 = torch.tensor(0.5, device=K.device) + integral_P2
    price = S0 * P1 - torch.exp(-r * T) * K * P2
    return price