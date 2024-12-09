import torch
from config import CONFIG

def fs_heston_cf(phi, S0, T0, T1, T2, r, kappa, v0, theta, sigma, rho):
    # Ensure that phi is a torch tensor on the GPU
    if not isinstance(phi, torch.Tensor):
        phi = torch.tensor(phi, dtype=torch.complex128, device=CONFIG.device)
    else:
        phi = phi.to(CONFIG.device).type(torch.complex128)


    S0 = S0.to(CONFIG.device).type(torch.float64)
    T0 = T0.to(CONFIG.device).type(torch.float64)
    T1 = T1.to(CONFIG.device).type(torch.float64)
    T2 = T2.to(CONFIG.device).type(torch.float64)
    r = r.to(CONFIG.device).type(torch.float64)

    tau = T2-T1

    delta = 4*kappa*theta/sigma**2
    little_c_bar = sigma**2/(4*kappa) * (1 - torch.exp(-kappa*(T1-T0)))
    kappa_bar = (4*kappa*v0*torch.exp(-kappa*(T1-T0))) / (sigma**2 * (1-torch.exp(-kappa*(T1-T0))))
    d = torch.sqrt((kappa-rho*sigma*1j*phi)**2 + sigma**2 * (phi**2 + 1j * phi))
    g = (kappa - rho*sigma*1j*phi-d)/(kappa-rho*sigma*1j*phi+d)

    A_bar = kappa*theta/sigma**2 * ((kappa - rho*sigma*1j*phi-d)*tau - 2*torch.log((1-g*torch.exp(-d*tau))/(1-g)))
    C_bar = (1-torch.exp(-d*tau))/(sigma**2 * (1-g*torch.exp(-d*tau))) * (kappa-rho*sigma*1j*phi - d)

    cf = torch.exp(A_bar + r*tau + (C_bar * little_c_bar*kappa_bar)/(1 - 2*C_bar*little_c_bar)) * (1/(1-2*C_bar*little_c_bar))**(delta/2)
    return cf


def fs_heston_price(S0, k, T0, T1, T2, r, kappa, v0, theta, sigma, rho):

    # Vérification de la taille de K et T
    assert k.dim() == 1, "K doit être un tenseur 1D"
    assert T0.dim() == 1, "T doit être un tenseur 1D"
    assert len(k) == len(T0), "K et T doivent avoir la même taille"


    umax = 50
    n = 100
    if n % 2 == 0:
        n += 1

    phi_values = torch.linspace(1e-5, umax, n, device=k.device)
    du = (umax - 1e-5) / (n - 1)

    phi_values = phi_values.unsqueeze(1).repeat(1, len(k))

    factor1 = torch.exp(-1j * phi_values * torch.log(k))
    denominator = 1j * phi_values


    cf1 = fs_heston_cf(phi_values - 1j, S0, T0, T1, T2, r, kappa, v0, theta, sigma, rho) / fs_heston_cf(-1j, S0, T0, T1, T2, r, kappa, v0, theta, sigma, rho)
    temp1 = factor1 * cf1 / denominator
    integrand_P1_values = 1 / torch.pi * torch.real(temp1)


    cf2 = fs_heston_cf(phi_values, S0, T0, T1, T2, r, kappa, v0, theta, sigma, rho)
    temp2 = factor1 * cf2 / denominator
    integrand_P2_values = 1 / torch.pi * torch.real(temp2)

    weights = torch.ones(n, device=k.device)
    weights[1:-1:2] = 4
    weights[2:-2:2] = 2
    weights *= du / 3
    weights = weights.unsqueeze(1).repeat(1, len(k))

    integral_P1 = torch.sum(weights * integrand_P1_values, dim=0)
    integral_P2 = torch.sum(weights * integrand_P2_values, dim=0)

    P1 = torch.tensor(0.5, device=k.device) + integral_P1
    P2 = torch.tensor(0.5, device=k.device) + integral_P2
    price = S0 * (P1 - torch.exp(-r * (T2-T1)) * k * P2)
    return price

S0 = torch.tensor(100, device=CONFIG.device)
k = torch.tensor([1], device=CONFIG.device)
T0 = torch.tensor([0], device=CONFIG.device)
T1 = torch.tensor([1], device=CONFIG.device)
T2 = torch.tensor([2], device=CONFIG.device)
r = torch.tensor(0.0, device=CONFIG.device)
kappa = torch.tensor(2, device=CONFIG.device)
v0 = torch.tensor(0.04, device=CONFIG.device)
theta = torch.tensor(0.04, device=CONFIG.device)
sigma = torch.tensor(0.2, device=CONFIG.device)
rho = torch.tensor(-0.7, device=CONFIG.device)



price = fs_heston_price(S0, k, T0, T1, T2, r, kappa, v0, theta, sigma, rho)
print(price)