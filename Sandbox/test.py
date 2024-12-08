import torch
from config import *

""" ----------------------- Characteristic Function --------------------"""


def ChFHestonModelForwardStartTorch(r, T1, T2, kappa, gamma, vbar, v0, rho):
    # Définition du type et device
    dtype = torch.complex128
    device = r.device

    # Conversion des scalaires en tenseurs complex128 si nécessaire
    r = r.to(device).type(dtype)
    T1 = T1.to(device).type(torch.float64)
    T2 = T2.to(device).type(torch.float64)
    kappa = kappa.to(device).type(torch.float64)
    gamma = gamma.to(device).type(torch.float64)
    vbar = vbar.to(device).type(torch.float64)
    v0 = v0.to(device).type(torch.float64)
    rho = rho.to(device).type(torch.float64)

    # Convertir en complex128 certains scalaires
    r = r.to(dtype)
    kappa = kappa.to(dtype)
    gamma = gamma.to(dtype)
    vbar = vbar.to(dtype)
    v0 = v0.to(dtype)
    rho = rho.to(dtype)
    T1 = T1.to(dtype)
    T2 = T2.to(dtype)

    i = torch.tensor([0.0+1.0j], dtype=dtype, device=device)[0]  # i = 1j
    tau = (T2 - T1).to(dtype)

    def D1(u):
        # D1(u) = sqrt((kappa - gamma*rho*i*u)^2 + (u^2 + i*u)*gamma^2)
        return torch.sqrt((kappa - gamma*rho*i*u)**2 + (u**2 + i*u)*gamma**2)

    def g(u):
        # g(u) = (kappa - gamma*rho*i*u - D1(u)) / (kappa - gamma*rho*i*u + D1(u))
        return (kappa - gamma*rho*i*u - D1(u)) / (kappa - gamma*rho*i*u + D1(u))

    def C(u):
        # C(u) = ((1 - exp(-D1(u)*tau)) / (gamma^2*(1 - g(u)*exp(-D1(u)*tau)))) * (kappa - gamma*rho*i*u - D1(u))
        return ( (1 - torch.exp(-D1(u)*tau)) / (gamma**2 * (1 - g(u)*torch.exp(-D1(u)*tau))) ) * (kappa - gamma*rho*i*u - D1(u))

    def A(u):
        # A(u) = r*i*u*tau + kappa*vbar*(tau/gamma^2)*(kappa - gamma*rho*i*u - D1(u)) - 2*kappa*vbar/gamma^2*log((1 - g(u)*exp(-D1(u)*tau)) / (1 - g(u)))
        return r*i*u*tau \
               + kappa*vbar*(tau/gamma**2)*(kappa - gamma*rho*i*u - D1(u)) \
               - 2*kappa*vbar/gamma**2*torch.log((1 - g(u)*torch.exp(-D1(u)*tau)) / (1 - g(u)))

    def c_bar(t1, t2):
        # c_bar = gamma^2/(4*kappa)*(1 - exp(-kappa*(t2-t1)))
        return gamma**2/(4*kappa)*(1 - torch.exp(-kappa*(t2-t1)))

    delta = 4*kappa*vbar/gamma**2

    def kappa_bar(t1, t2):
        # kappa_bar = 4*kappa*v0*exp(-kappa*(t2-t1)) / (gamma^2*(1 - exp(-kappa*(t2-t1))))
        return 4*kappa*v0*torch.exp(-kappa*(t2-t1)) / (gamma**2*(1 - torch.exp(-kappa*(t2-t1))))

    def term1(u):
        # term1(u) = A(u) + C(u)*c_bar(0,T1)*kappa_bar(0,T1)/(1 - 2*C(u)*c_bar(0,T1))
        numerator = C(u)*c_bar(0.0, T1)*kappa_bar(0.0, T1)
        denominator = 1 - 2*C(u)*c_bar(0.0, T1)
        return A(u) + numerator/denominator

    def term2(u):
        # term2(u) = (1/(1 - 2*C(u)*c_bar(0,T1)))^(0.5*delta)
        return torch.pow(1/(1 - 2*C(u)*c_bar(0.0, T1)), 0.5*delta)

    def cf(u):
        # cf(u) = exp(term1(u))*term2(u)
        return torch.exp(term1(u))*term2(u)

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


    cf1 = heston_cf(phi_values, S0, T, r, kappa, v0, theta, sigma, rho)
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
    price = S0 * (P1 - torch.exp(-r * T) * K * P2)
    return price

Kappa = torch.tensor(2, device=CONFIG.device)
v0 = torch.tensor(0.04, device=CONFIG.device)
Theta = torch.tensor(0.04, device=CONFIG.device)
sigma = torch.tensor(0.3, device=CONFIG.device)
rho = torch.tensor(-0.7, device=CONFIG.device)
k=torch.tensor([1.2], device=CONFIG.device)
S0 = torch.tensor(100, device=CONFIG.device)  # Initial stock price
r = torch.tensor(0.05, device=CONFIG.device)
T = torch.tensor([1], device=CONFIG.device)

print(heston_price(S0, k, T, r, Kappa, v0, Theta, sigma, rho))